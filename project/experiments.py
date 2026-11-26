"""Resumable evaluation of the nine frozen checkpoints.

No training is performed. Every invocation locks data, checkpoints, source code,
sample selection and attack settings in protocol.json. Use a separate output
directory for smoke tests. Completed and interrupted batches are reused only
under the identical protocol. Run core before extended.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from attacks import attack_batch, candidate_state
from wafer_model import CLASS_NAMES, NONE_CLASS, WaferCNN, image_batch, metrics_from_predictions, seed_everything

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
EPSILON = 8 / 255
METHODS = ("clean", "noise", "pgd")
SEEDS = (77, 78, 79)
CONDITIONS = ("clean", "fgsm8", "pgd20", "pgd50", "targeted_none")
PHASES = {"core": CONDITIONS[:3], "extended": ("pgd50",), "targeted": ("targeted_none",), "all": CONDITIONS}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def array_hash(value):
    return hashlib.sha256(np.asarray(value, dtype="<i8").tobytes()).hexdigest()


@contextmanager
def atomic_file(path, binary=False):
    """Write and fsync a sibling temporary file before one atomic replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("wb" if binary else "w", **({} if binary else {"encoding": "utf-8", "newline": ""})) as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json(path, value):
    with atomic_file(path) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_npz(path, **arrays):
    with atomic_file(path, binary=True) as stream:
        np.savez_compressed(stream, **arrays)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@contextmanager
def output_lock(output):
    """OS-held lock is released even if the evaluator process is terminated."""
    output.mkdir(parents=True, exist_ok=True)
    stream = (output / ".writer.lock").open("a+b")
    try:
        if stream.seek(0, os.SEEK_END) == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        stream.close()


def verify_attack(clean, attacked, epsilon):
    if clean.shape != attacked.shape or not bool(torch.isfinite(attacked).all()):
        raise ValueError("Attack shape or finite-value check failed")
    delta = attacked - clean
    maximum = float(delta.abs().max().item())
    background_unchanged = bool((delta[clean == 0] == 0).all())
    if maximum > epsilon + 1e-6 or not background_unchanged:
        raise ValueError("Attack exceeds L-infinity bound or changes fixed background")
    if float(attacked.min()) < 0 or float(attacked.max()) > 1:
        raise ValueError("Attack outside [0,1]")
    return {"max_linf": maximum, "background_unchanged": background_unchanged, "within_bounds": True}


def targeted_none_attack(model, clean, *, epsilon=EPSILON, steps=20, restarts=5, seed=0, target=NONE_CLASS):
    """Minimize CE(target); retain target successes before target objective.

    Success is prediction == target, never merely prediction != true class.
    Clean input, random starts, all visited iterates and final iterates compete.
    """
    if not np.isfinite(epsilon) or epsilon < 0 or steps < 1 or restarts < 1:
        raise ValueError("Invalid targeted attack settings")
    clean = clean.detach()
    mask = clean > 0
    targets = torch.full((len(clean),), target, dtype=torch.long, device=clean.device)
    with torch.no_grad():
        logits = model(clean)
        best_score = -F.cross_entropy(logits, targets, reduction="none")
        best_conf, best_pred = logits.softmax(1).max(1)
        best_x = clean.clone()

    @torch.no_grad()
    def retain(current, logits=None):
        nonlocal best_x, best_score, best_pred, best_conf
        logits = model(current) if logits is None else logits.detach()
        score = -F.cross_entropy(logits, targets, reduction="none")
        confidence, predicted = logits.softmax(1).max(1)
        success, best_success = predicted.eq(target), best_pred.eq(target)
        choose = (success & ~best_success) | (success.eq(best_success) & score.gt(best_score))
        best_x[choose] = current.detach()[choose]
        best_score[choose] = score[choose]
        best_pred[choose] = predicted[choose]
        best_conf[choose] = confidence[choose]

    generator = torch.Generator(device=clean.device).manual_seed(seed)
    for _ in range(restarts):
        noise = torch.empty_like(clean).uniform_(-epsilon, epsilon, generator=generator)
        current = (clean + noise * mask).clamp(0, 1)
        for _ in range(steps):
            current = current.detach().requires_grad_(True)
            logits = model(current)
            retain(current, logits)
            loss = F.cross_entropy(logits, targets, reduction="sum")
            gradient, = torch.autograd.grad(loss, current)
            with torch.no_grad():
                delta = (current - epsilon / 4 * gradient.sign() - clean).clamp(-epsilon, epsilon)
                current = (clean + delta * mask).clamp(0, 1)
        retain(current)
    verify_attack(clean, best_x, epsilon)
    return best_x.detach(), best_pred.detach(), best_conf.detach()


def targeted_metrics(labels, predictions, clean_predictions):
    labels, predictions, clean_predictions = map(np.asarray, (labels, predictions, clean_predictions))
    if np.any(labels == NONE_CLASS):
        raise ValueError("Targeted-to-None denominator must contain only true defects")
    success = predictions == NONE_CLASS
    clean_correct = clean_predictions == labels
    initial_none = clean_predictions == NONE_CLASS
    if np.any(initial_none & ~success):
        raise ValueError("Targeted attack lost a clean target success")
    denominator = int(clean_correct.sum())
    return {
        "target_class": "None", "target_class_id": NONE_CLASS,
        "target_success_count": int(success.sum()), "target_success_denominator": len(labels),
        "target_success_rate": float(success.mean()),
        "clean_correct_defect_success_count": int((success & clean_correct).sum()),
        "clean_correct_defect_denominator": denominator,
        "clean_correct_defect_success_rate": float(success[clean_correct].mean()) if denominator else None,
        "baseline_clean_defect_to_none_count": int(initial_none.sum()),
        "baseline_clean_defect_to_none_rate": float(initial_none.mean()),
        "new_target_success_count": int((success & ~initial_none).sum()),
    }


def select_indices(labels, test, smoke):
    test = np.sort(np.asarray(test, dtype=np.int64))
    if len(np.unique(test)) != len(test):
        raise ValueError("Duplicate test indices")
    if smoke:
        return np.sort(np.concatenate([test[labels[test] == c][:2] for c in range(len(CLASS_NAMES))]))
    return test


def checkpoints():
    manifest = read_json(ROOT / "models/manifest.json")
    runs = []
    for record in manifest["models"]:
        checkpoint = ROOT / record["path"]
        actual = sha256(checkpoint)
        if actual != record["new_container_sha256"]:
            raise ValueError("Public checkpoint differs from its manifest")
        runs.append({"method": record["method"], "seed": record["seed"],
                     "checkpoint": record["path"], "checkpoint_sha256": actual})
    if {(x["method"], x["seed"]) for x in runs} != {(m, s) for m in METHODS for s in SEEDS} or len(runs) != 9:
        raise ValueError("Expected exactly nine original method/seed checkpoints")
    return runs


def make_protocol(args, labels, selected, data_hashes, runs):
    smoke = args.smoke
    definition = {
        "schema_version": SCHEMA_VERSION, "scope": "smoke" if smoke else "full_test",
        "smoke_warning": "Short 2-step/1-restart software check; not scientific results" if smoke else None,
        "data_sha256": data_hashes, "runs": runs,
        "code_sha256": {name: sha256(ROOT / name) for name in ["project/experiments.py", "project/attacks.py", "project/wafer_model.py"]},
        "selected_indices_sha256": array_hash(selected), "n": len(selected),
        "class_counts": {name: int(np.sum(labels[selected] == c)) for c, name in enumerate(CLASS_NAMES)},
        "targeted_n": int(np.sum(labels[selected] != NONE_CLASS)), "batch_size": args.batch_size,
        "sample_order": "ascending frozen cache_index; smoke first two test maps per class",
        "epsilon": EPSILON, "epsilon_255": 8, "pgd_step_size": EPSILON / 4,
        "pgd20_steps": 2 if smoke else 20, "pgd50_steps": 2 if smoke else 50,
        "targeted_steps": 2 if smoke else 20, "restarts": 1 if smoke else 5,
        "attack_seed": "7701000 + 8*10000 + batch_start; shared PGD20/PGD50 starts",
        "untargeted_selection": "Clean plus all visited candidates; misclassification first, then greatest unweighted CE(true label)",
        "targeted_selection": "True defects only; clean plus all visited candidates; prediction None first, then smallest CE(None)",
        "threat_model": "White-box continuous post-preprocessing L-infinity perturbations within fixed clean wafer mask; values clipped to [0,1]",
        "input_encoding": "original 0/1/2 uint8 divided by 2; continuous digital threat, not physical die flips",
        "checkpoint_selection": "Reuse all nine frozen checkpoints selected by clean validation macro-F1; no retraining or test-based selection",
        "device": str(args.device), "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda, "numpy": np.__version__,
        "precision": "float32; no AMP or TF32; cuDNN deterministic; 4 CPU threads",
        "aggregation": "Mean and sample SD (ddof=1) over complete sets of training seeds 77,78,79 on one fixed lot split",
    }
    return {"protocol_sha256": object_hash(definition), "definition": definition}


def lock_protocol(output, protocol):
    path = output / "protocol.json"
    if path.exists():
        saved = read_json(path)
        if saved.get("protocol_sha256") != protocol["protocol_sha256"] or saved.get("definition") != protocol["definition"]:
            raise ValueError("Output protocol differs: use a new output directory; refusing to mix data/code/settings")
        return saved
    if any(output.glob("runs/*/*/metrics.json")) or any(output.glob("runs/*/*/batches/*.npz")):
        raise ValueError("Existing experiment artifacts have no protocol; refusing reuse")
    saved = {**protocol, "created_at_utc": utc_now()}
    write_json(path, saved)
    return saved


def batch_path(directory, start):
    return directory / "batches" / f"batch_{start:07d}.npz"


def load_batch(path, config_hash, indices, labels):
    with np.load(path, allow_pickle=False) as stored:
        result = {name: np.array(stored[name]) for name in stored.files}
    if str(result["config_sha256"].item()) != config_hash:
        raise ValueError(f"Batch config mismatch: {path}")
    for key, expected in [("cache_index", indices), ("true_class_id", labels)]:
        if not np.array_equal(result[key], expected):
            raise ValueError(f"Batch {key} mismatch: {path}")
    for key in ("predicted_id", "clean_predicted_id"):
        value = result[key]
        if value.shape != indices.shape or not np.issubdtype(value.dtype, np.integer) or np.any((value < 0) | (value >= len(CLASS_NAMES))):
            raise ValueError(f"Invalid batch {key}: {path}")
    confidence = result["confidence"]
    if confidence.shape != indices.shape or not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
        raise ValueError(f"Invalid batch confidence: {path}")
    if float(result["max_linf"]) > EPSILON + 1e-6 or not bool(result["background_unchanged"]):
        raise ValueError(f"Invalid batch invariants: {path}")
    return result


def aggregate_runs(runs):
    aggregates = []
    for method in METHODS:
        for condition in CONDITIONS:
            entries = sorted([x for x in runs if x["method"] == method and x["condition"] == condition], key=lambda x: x["seed"])
            if [x["seed"] for x in entries] != list(SEEDS):
                continue
            if len({x["n"] for x in entries}) != 1:
                raise ValueError("Unequal sample counts across seeds")
            def stats(values):
                if any(x is None for x in values):
                    return {"mean": None, "sd": None}
                return {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1))}
            names = ["accuracy", "macro_f1", "defect_to_none_rate"]
            if condition == "targeted_none":
                names += ["target_success_rate", "clean_correct_defect_success_rate", "baseline_clean_defect_to_none_rate"]
            aggregates.append({"method": method, "condition": condition, "n": entries[0]["n"], "seeds": list(SEEDS),
                               "metrics": {name: stats([x["metrics"][name] for x in entries]) for name in names},
                               "per_class_recall": {name: stats([x["metrics"]["per_class_recall"][name] for x in entries]) for name in CLASS_NAMES}})
    return aggregates


def update_summary(output, protocol):
    runs = []
    for record in protocol["definition"]["runs"]:
        for condition in CONDITIONS:
            directory = output / "runs" / f'{record["method"]}_seed{record["seed"]}' / condition
            path = directory / "metrics.json"
            if not path.exists():
                continue
            result = read_json(path)
            if result["protocol_sha256"] != protocol["protocol_sha256"]:
                raise ValueError(f"Mixed protocol in {path}")
            predictions = directory / "predictions.csv"
            if sha256(predictions) != result["predictions_sha256"]:
                raise ValueError(f"Prediction digest differs: {predictions}")
            runs.append({"method": record["method"], "seed": record["seed"], "condition": condition, "n": result["metrics"]["n"],
                         "scope": result["scope"], "metrics": result["metrics"], "seconds": result["seconds"],
                         "predictions_path": predictions.relative_to(output).as_posix(), "metrics_path": path.relative_to(output).as_posix()})
    complete_conditions = [c for c in CONDITIONS if sum(x["condition"] == c for x in runs) == 9]
    summary = {"schema_version": SCHEMA_VERSION, "scope": protocol["definition"]["scope"], "protocol_sha256": protocol["protocol_sha256"],
               "updated_at_utc": utc_now(), "complete_conditions": complete_conditions, "complete": len(runs) == 45,
               "completed_runs": len(runs), "expected_runs": 45, "runs": runs, "aggregate": aggregate_runs(runs),
               "statistical_scope": protocol["definition"]["aggregation"],
               "smoke_warning": protocol["definition"]["smoke_warning"]}
    write_json(output / "summary.json", summary)
    return summary


def progress(output, **fields):
    value = {"updated_at_utc": utc_now(), "pid": os.getpid(), **fields}
    write_json(output / "progress.json", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def evaluate_condition(output, protocol, run, condition, model, images, labels, selected, device):
    settings = protocol["definition"]
    if condition == "targeted_none":
        selected = selected[labels[selected] != NONE_CLASS]
    directory = output / "runs" / f'{run["method"]}_seed{run["seed"]}' / condition
    config = {"protocol_sha256": protocol["protocol_sha256"], "method": run["method"], "seed": run["seed"],
              "condition": condition, "checkpoint_sha256": run["checkpoint_sha256"], "selected_indices_sha256": array_hash(selected),
              "n": len(selected)}
    config_hash = object_hash(config)
    config_path = directory / "config.json"
    if config_path.exists() and read_json(config_path) != config:
        raise ValueError(f"Condition config changed: {config_path}")
    if not config_path.exists():
        write_json(config_path, config)
    completed_path = directory / "metrics.json"
    if completed_path.exists():
        completed = read_json(completed_path)
        if completed["config_sha256"] != config_hash or sha256(directory / "predictions.csv") != completed["predictions_sha256"]:
            raise ValueError(f"Completed artifact changed: {directory}")
        for filename, digest in completed["batch_sha256"].items():
            if sha256(directory / filename) != digest:
                raise ValueError(f"Completed batch changed: {directory / filename}")
        progress(output, status="condition_reused", method=run["method"], seed=run["seed"], condition=condition, completed=len(selected), total=len(selected))
        return
    rows, batch_hashes, elapsed = [], {}, 0.0
    max_linf = 0.0
    condition_started = time.perf_counter()
    core_directory = directory.parent / "pgd20"
    if condition == "pgd50" and not (core_directory / "metrics.json").exists():
        raise ValueError("PGD50 requires completed PGD20 for the same frozen samples; run --phase core first")
    core_config_hash = object_hash(read_json(core_directory / "config.json")) if condition == "pgd50" else None
    for start in range(0, len(selected), settings["batch_size"]):
        indices = selected[start:start + settings["batch_size"]]
        truth = np.asarray(labels[indices], dtype=np.int64)
        path = batch_path(directory, start)
        if path.exists():
            batch = load_batch(path, config_hash, indices, truth)
        else:
            begin = time.perf_counter()
            x = image_batch(images, indices, device)
            y = torch.from_numpy(truth).to(device)
            _, clean_pred, clean_conf = candidate_state(model, x, y)
            seed = 7701000 + 8 * 10000 + start
            if start == 0:
                zero, zero_pred, _ = attack_batch(model, x[:2], y[:2], 0, "fgsm")
                target_zero, target_zero_pred, _ = targeted_none_attack(model, x[:2], epsilon=0, steps=1, restarts=1)
                if not torch.equal(zero, x[:2]) or not torch.equal(target_zero, x[:2]) or not torch.equal(zero_pred, clean_pred[:2]) or not torch.equal(target_zero_pred, clean_pred[:2]):
                    raise ValueError("Epsilon-zero identity failed")
            if condition == "clean":
                adv, pred, confidence = x, clean_pred, clean_conf
            elif condition == "targeted_none":
                adv, pred, confidence = targeted_none_attack(model, x, steps=settings["targeted_steps"], restarts=settings["restarts"], seed=seed)
            else:
                kind = "fgsm" if condition == "fgsm8" else "pgd"
                steps = settings["pgd50_steps"] if condition == "pgd50" else settings["pgd20_steps"]
                adv, pred, confidence = attack_batch(model, x, y, EPSILON, kind, steps=steps, restarts=settings["restarts"], seed=seed)
            invariants = verify_attack(x, adv, 0 if condition == "clean" else EPSILON)
            predicted = pred.cpu().numpy()
            clean_prediction = clean_pred.cpu().numpy()
            if condition == "targeted_none":
                targeted_metrics(truth, predicted, clean_prediction)
            elif np.any((clean_prediction != truth) & (predicted == truth)):
                raise ValueError("Untargeted attack repaired a clean error despite clean-candidate retention")
            batch = {"config_sha256": np.array(config_hash), "cache_index": indices, "true_class_id": truth,
                     "predicted_id": predicted, "confidence": confidence.cpu().numpy(), "clean_predicted_id": clean_prediction,
                     "clean_confidence": clean_conf.cpu().numpy(), "seconds": np.array(time.perf_counter() - begin),
                     "max_linf": np.array(invariants["max_linf"]), "background_unchanged": np.array(True)}
            write_npz(path, **batch)
        if condition == "pgd50":
            previous = load_batch(batch_path(core_directory, start), core_config_hash, indices, truth)
            if np.any((previous["predicted_id"] != truth) & (batch["predicted_id"] == truth)):
                raise ValueError("PGD50 lost PGD20 failures under shared random starts")
        rows.append(pd.DataFrame({name: batch[name] for name in ("cache_index", "true_class_id", "predicted_id", "confidence", "clean_predicted_id", "clean_confidence")}))
        elapsed += float(batch["seconds"])
        max_linf = max(max_linf, float(batch["max_linf"]))
        batch_hashes[path.relative_to(directory).as_posix()] = sha256(path)
        if start == 0 or (start // settings["batch_size"] + 1) % 25 == 0 or start + len(indices) == len(selected):
            progress(output, status="running", method=run["method"], seed=run["seed"], condition=condition,
                     completed=start + len(indices), total=len(selected), measured_batch_seconds=elapsed)
    table = pd.concat(rows, ignore_index=True)
    if not np.array_equal(table.cache_index, selected):
        raise ValueError("Final predictions do not cover ordered evaluation samples exactly")
    metrics = metrics_from_predictions(table.true_class_id, table.predicted_id)
    if condition == "targeted_none":
        metrics.update(targeted_metrics(table.true_class_id, table.predicted_id, table.clean_predicted_id))
    with atomic_file(directory / "predictions.csv") as stream:
        table.to_csv(stream, index=False, lineterminator="\n")
    scope = settings["scope"] + ("_defects" if condition == "targeted_none" else "")
    result = {"schema_version": SCHEMA_VERSION, "method": run["method"], "seed": run["seed"], "condition": condition,
              "scope": scope, "config": config, "config_sha256": config_hash, "protocol_sha256": protocol["protocol_sha256"],
              "checkpoint_sha256": run["checkpoint_sha256"], "data_sha256": settings["data_sha256"], "code_sha256": settings["code_sha256"],
              "metrics": metrics, "seconds": elapsed, "invocation_seconds": time.perf_counter() - condition_started,
              "timing_scope": "Sum of persisted batch wall times, including transfers, attack generation and checks; not isolated hardware benchmark",
              "predictions_sha256": sha256(directory / "predictions.csv"), "batch_sha256": batch_hashes,
              "checks": {"epsilon_zero_identity": True, "fixed_background": True, "max_linf": max_linf,
                         "values_in_zero_one": True, "clean_candidate_retained": True,
                         "pgd50_preserves_pgd20_failures": True if condition == "pgd50" else None},
              "finished_at_utc": utc_now()}
    write_json(completed_path, result)
    update_summary(output, protocol)
    progress(output, status="condition_complete", method=run["method"], seed=run["seed"], condition=condition,
             completed=len(selected), total=len(selected), seconds=elapsed, metrics=metrics)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/reproduction")
    parser.add_argument("--phase", choices=PHASES, default="all")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if ROOT / "results" not in output.parents:
        parser.error("Use a dedicated output directory underneath results/")
    if not args.smoke and args.device != "cuda":
        parser.error("Full evaluation requires CUDA; CPU is permitted only for --smoke")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent CPU fallback")
    with output_lock(output):
        return execute(args, output)


def execute(args, output):
    seed_everything(7701)
    data = ROOT / "data/processed"
    images = np.load(data / "images.npy", mmap_mode="r")
    labels = np.load(data / "labels.npy", mmap_mode="r")
    with np.load(data / "splits.npz", allow_pickle=False) as stored:
        test = np.array(stored["test"], dtype=np.int64)
    if images.shape != (len(labels), 64, 64) or images.dtype != np.uint8:
        raise ValueError("Unexpected input cache")
    if len(test) != 25485 or np.sum(labels[test] != NONE_CLASS) != 3753:
        raise ValueError("Expected original full test 25,485 maps and 3,753 true defects")
    selected = select_indices(labels, test, args.smoke)
    data_hashes = {name: sha256(data / name) for name in ("images.npy", "labels.npy", "splits.npz", "metadata.csv")}
    runs = checkpoints()
    protocol = lock_protocol(output, make_protocol(args, labels, selected, data_hashes, runs))
    write_json(output / "selection.json", {"scope": protocol["definition"]["scope"], "cache_indices": selected.tolist(), "sha256": array_hash(selected)})
    update_summary(output, protocol)
    device = torch.device(args.device)
    for condition in PHASES[args.phase]:
        for run in runs:
            checkpoint = torch.load(ROOT / run["checkpoint"], map_location=device, weights_only=False)
            if checkpoint["config"]["data_sha256"] != data_hashes:
                raise ValueError("Checkpoint does not match frozen processed dataset")
            model = WaferCNN().to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            evaluate_condition(output, protocol, run, condition, model, images, labels, selected, device)
            del model, checkpoint
    summary = update_summary(output, protocol)
    progress(output, status="phase_complete", phase=args.phase, scope=protocol["definition"]["scope"],
             completed_runs=summary["completed_runs"], complete_conditions=summary["complete_conditions"])
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__, "error": str(error), "updated_at_utc": utc_now()}), file=sys.stderr, flush=True)
        raise
