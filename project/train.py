"""Train the CNN on frozen lot-disjoint splits; select only by validation F1.

No model choice uses test results. Each invocation trains one seed and mode;
aggregate independent invocations for the three-seed comparison. --no-test postpones test
evaluation; --evaluate-only later evaluates the already selected best checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from wafer_model import CLASS_NAMES, WaferCNN, image_batch, metrics_from_predictions, predict, seed_everything
from training_defenses import pgd_training_examples

from artifact_paths import relative_path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_json(value, path):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def save_checkpoint(value, path):
    temporary = Path(path).with_suffix(".pt.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def plot_learning(history, out):
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(df.epoch, df.train_loss, label="Train", color="#386f94")
    axes[0].plot(df.epoch, df.val_loss, label="Validation", color="#bf7044")
    axes[0].set(xlabel="Epoch", ylabel="Weighted cross-entropy", title="Optimization history")
    axes[0].legend(frameon=False)
    axes[1].plot(df.epoch, df.val_accuracy, label="Accuracy", color="#386f94")
    axes[1].plot(df.epoch, df.val_macro_f1, label="Macro-F1", color="#bf7044")
    axes[1].set(xlabel="Epoch", ylabel="Score", ylim=(0, 1), title="Validation performance")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=220)
    plt.close(fig)


def save_evaluation(name, result, preds, conf, indices, labels, metadata, out):
    cm = np.array(result["confusion_matrix"])
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(out / f"{name}_confusion_matrix.csv", encoding="utf-8-sig", index_label="true_class")
    normalized = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(9), CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(9), CLASS_NAMES)
    ax.set(xlabel="Predicted class", ylabel="True class", title=f"{name.capitalize()} confusion matrix (row normalized)")
    for row in range(9):
        for col in range(9):
            value = normalized[row, col]
            ax.text(col, row, f"{100*value:.1f}", ha="center", va="center", fontsize=8, color="white" if value > .55 else "black")
    fig.colorbar(image, ax=ax, label="Fraction of true class", shrink=.8)
    fig.tight_layout()
    fig.savefig(out / f"{name}_confusion_matrix.png", dpi=220)
    plt.close(fig)
    rows = [{"class": c, "support": result["per_class_support"][c], "recall": result["per_class_recall"][c]} for c in CLASS_NAMES]
    pd.DataFrame(rows).to_csv(out / f"{name}_per_class.csv", index=False, encoding="utf-8-sig")
    selected = metadata.iloc[np.asarray(indices)].copy()
    selected["true_class_id"] = labels[indices]
    selected["true_class"] = [CLASS_NAMES[int(v)] for v in labels[indices]]
    selected["predicted_class_id"] = preds
    selected["predicted_class"] = [CLASS_NAMES[int(v)] for v in preds]
    selected["confidence"] = conf
    keep = [c for c in ["cache_index", "original_row", "original_dataframe_index", "lot", "true_class_id", "true_class", "predicted_class_id", "predicted_class", "confidence"] if c in selected]
    selected[keep].to_csv(out / f"{name}_predictions.csv", index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--mode", choices=["clean", "noise", "pgd"], default="clean")
    parser.add_argument("--noise-epsilon", type=float, default=8 / 255)
    parser.add_argument("--pgd-epsilon", type=float, default=8 / 255)
    parser.add_argument("--pgd-steps", type=int, default=7)
    parser.add_argument("--pgd-step-size", type=float, default=2 / 255)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--no-test", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise ValueError("Epochs, patience, and batch size must be positive.")
    if (not np.isfinite(args.noise_epsilon) or args.noise_epsilon < 0 or
            not np.isfinite(args.pgd_epsilon) or args.pgd_epsilon < 0 or
            not np.isfinite(args.pgd_step_size) or args.pgd_step_size <= 0 or args.pgd_steps < 1):
        raise ValueError("Perturbation bounds must be nonnegative; PGD steps and step size must be positive and finite.")
    started = time.perf_counter()
    prefix = "baseline" if args.mode == "clean" else args.mode
    run = (args.run_dir or ROOT / "results" / "training" / f"{prefix}_seed{args.seed}").resolve()
    run.mkdir(parents=True, exist_ok=True)
    if not args.evaluate_only and any((run / n).exists() for n in ["best.pt", "history.csv", "config.json"]):
        raise FileExistsError("This run directory already contains training artifacts. Use a new --run-dir, or --evaluate-only.")
    seed_everything(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing an unannounced CPU fallback.")
    device = torch.device(args.device)
    data = args.data_dir.resolve()
    images = np.load(data / "images.npy", mmap_mode="r")
    labels = np.load(data / "labels.npy", mmap_mode="r")
    with np.load(data / "splits.npz") as stored:
        split = {name: np.array(stored[name], dtype=np.int64) for name in ["train", "val", "test"]}
    if images.dtype != np.uint8 or images.shape[1:] != (64, 64):
        raise ValueError("Expected uint8 images with shape (N, 64, 64).")
    if len(images) != len(labels):
        raise ValueError("Images and labels differ in length.")
    combined = np.concatenate(list(split.values()))
    if len(combined) != len(labels) or not np.array_equal(np.sort(combined), np.arange(len(labels))):
        raise ValueError("Splits must partition every cache index exactly once.")
    counts = np.bincount(labels[split["train"]], minlength=9)
    if len(counts) != 9 or np.any(counts == 0):
        raise ValueError("Training split must contain all nine classes.")
    weights = 1 / np.sqrt(counts.astype(np.float64))
    weights /= weights.mean()
    weights_tensor = torch.tensor(weights, device=device, dtype=torch.float32)
    fingerprints = {n: digest(data / n) for n in ["images.npy", "labels.npy", "splits.npz", "metadata.csv"]}
    model = WaferCNN().to(device)
    metadata = pd.read_csv(data / "metadata.csv")
    if "cache_index" not in metadata or not np.array_equal(metadata.cache_index.to_numpy(), np.arange(len(labels))):
        raise ValueError("metadata.csv must be in cache-index order.")
    if args.evaluate_only:
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        if config["data_sha256"] != fingerprints:
            raise ValueError("Processed dataset differs from the checkpoint's frozen data.")
        checkpoint = torch.load(run / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        training_summary = json.loads((run / "training_summary.json").read_text(encoding="utf-8"))
        history = pd.read_csv(run / "history.csv").to_dict("records")
    else:
        config = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "data_dir": relative_path(data),
            "data_sha256": fingerprints, "class_names": CLASS_NAMES,
            "seed": args.seed, "training_mode": args.mode, "max_epochs": args.epochs,
            "early_stopping_patience": args.patience, "selection_metric": "validation_macro_f1_fixed_nine_classes",
            "selection_tie_rule": "Keep the earliest epoch on equal validation macro-F1.",
            "batch_size": args.batch_size, "eval_batch_size": args.eval_batch_size,
            "optimizer": "Adam", "learning_rate": args.learning_rate, "weight_decay": 0.0,
            "train_class_counts": counts.tolist(), "class_weights": weights.tolist(),
            "weight_rule": "inverse sqrt training class count, normalized to arithmetic mean 1",
            "loss": "weighted cross entropy", "noise_epsilon": args.noise_epsilon if args.mode == "noise" else None,
            "noise_rule": "uniform[-epsilon,epsilon] on the clean wafer mask, clip to [0,1]" if args.mode == "noise" else None,
            "pgd_training": {
                "epsilon": args.pgd_epsilon, "steps": args.pgd_steps, "step_size": args.pgd_step_size,
                "restarts": 1, "random_start": "uniform[-epsilon,epsilon] on the fixed clean wafer mask",
                "projection": "after every step: clean +/- epsilon intersect [0,1], background fixed",
                "inner_objective": "unweighted cross entropy; last PGD iterate, detached",
                "outer_objective": "0.5 * weighted clean CE + 0.5 * weighted adversarial CE",
                "clean_loss_weight": 0.5, "adversarial_loss_weight": 0.5,
                "model_mode": "train; CNN has no batch normalization or dropout",
            } if args.mode == "pgd" else None,
            "augmentation": {"clean": "none", "noise": "uniform masked input noise", "pgd": "masked PGD adversarial training"}[args.mode],
            "input_encoding": "cached uint8 0/1/2 converted to float32 0/0.5/1",
            "model": "3 x (Conv3x3 stride1 padding1, ReLU, MaxPool2x2): channels 32/64/128; GAP; Linear128to9",
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "numpy": np.__version__, "precision": "float32; no AMP or TF32",
            "determinism": "fixed Python/NumPy/Torch seeds; cuDNN deterministic true, benchmark false",
            "shuffle": "NumPy default_rng(seed).permutation(train indices), once per epoch",
            "split_sizes": {k: len(v) for k, v in split.items()},
            "scope": "One independent seed/mode run; report aggregate only after the prescribed seed set is complete.",
            "training_code_sha256": {name: digest(ROOT / "project" / name)
                                     for name in ["train.py", "training_defenses.py", "wafer_model.py"]},
        }
        save_json(config, run / "config.json")
        print(json.dumps({"run": str(run), "mode": args.mode, "split_sizes": config["split_sizes"], "train_class_counts": counts.tolist(), "parameters": config["parameter_count"]}), flush=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        rng = np.random.default_rng(args.seed)
        best_score, best_epoch, stale = -1.0, 0, 0
        history = []
        train_start = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.perf_counter()
            model.train()
            ordering = rng.permutation(split["train"])
            loss_numerator = 0.0
            loss_denominator = 0.0
            clean_loss_numerator = 0.0
            adversarial_loss_numerator = 0.0
            batches = (len(ordering) + args.batch_size - 1) // args.batch_size
            for batch_no, start in enumerate(range(0, len(ordering), args.batch_size), 1):
                batch = ordering[start:start + args.batch_size]
                x = image_batch(images, batch, device)
                y = torch.from_numpy(np.array(labels[batch], dtype=np.int64)).to(device)
                if args.mode == "noise":
                    mask = x > 0
                    noise = torch.empty_like(x).uniform_(-args.noise_epsilon, args.noise_epsilon)
                    x = torch.clamp(x + noise * mask, 0, 1)
                optimizer.zero_grad(set_to_none=True)
                if args.mode == "pgd":
                    adversarial = pgd_training_examples(model, x, y, epsilon=args.pgd_epsilon,
                                                        steps=args.pgd_steps, step_size=args.pgd_step_size)
                    clean_loss = nn.functional.cross_entropy(model(x), y, weight=weights_tensor)
                    adversarial_loss = nn.functional.cross_entropy(model(adversarial), y, weight=weights_tensor)
                    loss = 0.5 * clean_loss + 0.5 * adversarial_loss
                else:
                    logits = model(x)
                    loss = nn.functional.cross_entropy(logits, y, weight=weights_tensor)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss.")
                loss.backward()
                optimizer.step()
                denominator = float(weights_tensor[y].sum().item())
                loss_numerator += float(loss.item()) * denominator
                loss_denominator += denominator
                if args.mode == "pgd":
                    clean_loss_numerator += float(clean_loss.item()) * denominator
                    adversarial_loss_numerator += float(adversarial_loss.item()) * denominator
                if batch_no % 250 == 0:
                    print(f"epoch {epoch}/{args.epochs}, batch {batch_no}/{batches}, elapsed {time.perf_counter()-epoch_start:.1f}s", flush=True)
            validation, _, _ = predict(model, images, labels, split["val"], device, args.eval_batch_size, weights_tensor)
            row = {"epoch": epoch, "train_loss": loss_numerator / loss_denominator,
                   "val_loss": validation["weighted_cross_entropy"], "val_accuracy": validation["accuracy"],
                   "val_macro_f1": validation["macro_f1"], "val_defect_to_none_rate": validation["defect_to_none_rate"],
                   "epoch_seconds": time.perf_counter() - epoch_start}
            if args.mode == "pgd":
                row["train_clean_loss"] = clean_loss_numerator / loss_denominator
                row["train_adversarial_loss"] = adversarial_loss_numerator / loss_denominator
            history.append(row)
            with (run / "history.csv").open("a", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                if epoch == 1:
                    writer.writeheader()
                writer.writerow(row)
            improved = validation["macro_f1"] > best_score
            if improved:
                best_score, best_epoch, stale = validation["macro_f1"], epoch, 0
                save_checkpoint({"model_state_dict": model.state_dict(), "config": config, "epoch": epoch,
                                 "validation_metrics": validation}, run / "best.pt")
            else:
                stale += 1
            save_checkpoint({"model_state_dict": model.state_dict(), "config": config, "epoch": epoch,
                             "optimizer_state_dict": optimizer.state_dict(), "rng_state": rng.bit_generator.state,
                             "torch_rng_state": torch.get_rng_state(),
                             "cuda_rng_state_all": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}, run / "last.pt")
            print(f"epoch {epoch}: train_loss={row['train_loss']:.6f}, val_loss={row['val_loss']:.6f}, val_accuracy={row['val_accuracy']:.6f}, val_macro_f1={row['val_macro_f1']:.6f}, seconds={row['epoch_seconds']:.1f}, best={best_epoch}", flush=True)
            if stale >= args.patience:
                print(f"Early stopping: no validation macro-F1 improvement for {args.patience} epochs.", flush=True)
                break
        training_summary = {"actual_epochs": len(history), "best_epoch": best_epoch,
                            "best_validation_macro_f1": best_score, "early_stopped": len(history) < args.epochs,
                            "training_seconds": time.perf_counter() - train_start,
                            "training_samples_per_epoch": len(split["train"]),
                            "note": "All training records used once per epoch; no test-based model selection."}
        save_json(training_summary, run / "training_summary.json")
        checkpoint = torch.load(run / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    plot_learning(history, run)
    majority = int(np.argmax(counts))
    all_metrics = {"config": config, "training": training_summary,
                   "best_checkpoint_sha256": digest(run / "best.pt"),
                   "majority_class_selected_from_training": CLASS_NAMES[majority],
                   "test_evaluated": not args.no_test, "macro_f1_labels": list(range(9)), "zero_division": 0}
    for name in ["val"] + ([] if args.no_test else ["test"]):
        metrics, preds, conf = predict(model, images, labels, split[name], device, args.eval_batch_size, weights_tensor)
        all_metrics[name] = metrics
        all_metrics[f"{name}_majority_baseline"] = metrics_from_predictions(labels[split[name]], np.full(len(split[name]), majority))
        save_evaluation(name, metrics, preds, conf, split[name], labels, metadata, run)
        print(f"{name}: accuracy={metrics['accuracy']:.6f}, macro_f1={metrics['macro_f1']:.6f}, defect_to_none={metrics['defect_to_none_rate']:.6f}", flush=True)
    all_metrics["invocation_seconds"] = time.perf_counter() - started
    all_metrics["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    save_json(all_metrics, run / "metrics.json")
    print(f"Completed; results saved in {run}", flush=True)


if __name__ == "__main__":
    main()
