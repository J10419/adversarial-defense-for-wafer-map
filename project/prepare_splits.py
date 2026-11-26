"""Clean WM-811K, cache 64x64 maps, and freeze lot-disjoint 70/15/15 splits.

All decisions use labels and map identity only, never model performance. The raw
pickle is immutable. Training must use only the returned train indices.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import struct
import time

import numpy as np
import pandas as pd
from PIL import Image
import PIL
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

from artifact_paths import relative_path

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Random", "Scratch", "Near-full", "None"]
CLASS_IDS = {name.casefold(): i for i, name in enumerate(CLASSES)}
EXCLUSION_FIELDS = ["original_row", "original_dataframe_index", "label", "lot", "stage", "reason", "representative_original_row", "identity_hash", "group_labels"]


def unwrap(value):
    while isinstance(value, (np.ndarray, list, tuple)):
        if isinstance(value, np.ndarray) and value.ndim == 0:
            value = value.item()
            continue
        if len(value) == 0:
            return None
        if len(value) != 1:
            return repr(value)
        value = value[0]
    if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value).strip() or None


def map_hash(array):
    array = np.asarray(array, dtype=np.uint8)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(struct.pack("<II", *array.shape))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw/LSWMD.pkl")
    parser.add_argument("--processed", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "data")
    parser.add_argument("--seed", type=int, default=77)
    args = parser.parse_args()
    start = time.perf_counter()
    processed, output = args.processed.resolve(), args.output.resolve()
    processed.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    print(f"Loading {args.input.resolve()}", flush=True)
    df = pd.read_pickle(args.input)
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The input must contain a pandas DataFrame")
    for column in ["waferMap", "failureType", "lotName"]:
        if column not in df:
            raise KeyError(column)
    print(f"Loaded {len(df):,} maps; indexing recognized labeled maps", flush=True)
    total = len(df)
    counts = Counter()
    initial_class_counts = Counter()
    original_groups = defaultdict(list)
    records = []
    raw_maps = df["waferMap"].array
    raw_labels = df["failureType"].array
    raw_lots = df["lotName"].array
    exclusion_path = output / "cleaning_exclusions.csv"
    with exclusion_path.open("w", newline="", encoding="utf-8-sig") as exclusion_stream:
        writer = csv.DictWriter(exclusion_stream, fieldnames=EXCLUSION_FIELDS)
        writer.writeheader()

        def exclude(record, stage, reason, representative="", digest="", labels=""):
            counts[reason] += 1
            writer.writerow({"original_row": record["original_row"],
                             "original_dataframe_index": record["original_dataframe_index"],
                             "label": record["label"], "lot": record["lot"],
                             "stage": stage, "reason": reason,
                             "representative_original_row": representative,
                             "identity_hash": digest, "group_labels": labels})

        for pos, raw in enumerate(raw_maps):
            label_text = unwrap(raw_labels[pos])
            class_id = CLASS_IDS.get(label_text.casefold()) if label_text is not None else None
            record = {"original_row": pos, "original_dataframe_index": str(df.index[pos]),
                      "label": CLASSES[class_id] if class_id is not None else label_text,
                      "class_id": class_id, "lot": unwrap(raw_lots[pos])}
            if label_text is None:
                exclude(record, "eligibility", "unlabeled")
                continue
            if class_id is None:
                exclude(record, "eligibility", "unexpected_label")
                continue
            initial_class_counts[CLASSES[class_id]] += 1
            if record["lot"] is None:
                exclude(record, "eligibility", "missing_lot")
                continue
            array = np.asarray(raw)
            if array.ndim != 2 or not array.size:
                exclude(record, "eligibility", "invalid_shape")
                continue
            if not np.isin(array, [0, 1, 2]).all():
                exclude(record, "eligibility", "invalid_pixel_values")
                continue
            if not np.any(array > 0):
                exclude(record, "eligibility", "no_valid_die")
                continue
            record.update({"raw_height": int(array.shape[0]), "raw_width": int(array.shape[1]),
                           "raw_hash": map_hash(array)})
            original_groups[record["raw_hash"]].append(len(records))
            records.append(record)
            if len(records) % 50_000 == 0:
                print(f"Indexed {len(records):,} labeled maps", flush=True)

        valid_labeled = len(records)
        raw_keep = []
        raw_conflict_groups = 0
        raw_duplicate_groups = 0
        for digest, members in original_groups.items():
            classes = {records[i]["class_id"] for i in members}
            group_labels = "|".join(CLASSES[i] for i in sorted(classes))
            if len(classes) > 1:
                raw_conflict_groups += 1
                for idx in members:
                    exclude(records[idx], "original_map", "raw_label_conflict", digest=digest, labels=group_labels)
                continue
            first = members[0]  # Source rows were traversed in ascending order.
            raw_keep.append(first)
            if len(members) > 1:
                raw_duplicate_groups += 1
                for idx in members[1:]:
                    exclude(records[idx], "original_map", "raw_same_label_duplicate", records[first]["original_row"], digest, group_labels)
        raw_keep.sort(key=lambda i: records[i]["original_row"])
        raw_retained = [records[i] for i in raw_keep]
        del original_groups, records, raw_keep
        gc.collect()
        print(f"Original-map cleaning retained {len(raw_retained):,}; resizing with Pillow NEAREST", flush=True)

        candidate_path = processed / "_resized_candidates.npy"
        candidates = np.lib.format.open_memmap(candidate_path, mode="w+", dtype=np.uint8,
                                               shape=(len(raw_retained), 64, 64))
        resized_groups = defaultdict(list)
        for idx, record in enumerate(raw_retained):
            source = np.asarray(raw_maps[record["original_row"]], dtype=np.uint8)
            array = np.asarray(Image.fromarray(source).resize((64, 64), Image.Resampling.NEAREST), dtype=np.uint8)
            if not np.any(array > 0):
                exclude(record, "resized_map", "resized_no_valid_die")
                continue
            assert np.isin(array, [0, 1, 2]).all()
            candidates[idx] = array
            record["resized_hash"] = map_hash(array)
            resized_groups[record["resized_hash"]].append(idx)
            if (idx + 1) % 50_000 == 0:
                print(f"Resized {idx+1:,}/{len(raw_retained):,}", flush=True)
        candidates.flush()
        keep_candidates = []
        resized_conflict_groups = 0
        resized_duplicate_groups = 0
        for digest, members in resized_groups.items():
            classes = {raw_retained[i]["class_id"] for i in members}
            group_labels = "|".join(CLASSES[i] for i in sorted(classes))
            if len(classes) > 1:
                resized_conflict_groups += 1
                for idx in members:
                    exclude(raw_retained[idx], "resized_map", "resized_label_conflict", digest=digest, labels=group_labels)
                continue
            first = members[0]
            keep_candidates.append(first)
            if len(members) > 1:
                resized_duplicate_groups += 1
                for idx in members[1:]:
                    exclude(raw_retained[idx], "resized_map", "resized_same_label_duplicate", raw_retained[first]["original_row"], digest, group_labels)

    keep_candidates.sort(key=lambda i: raw_retained[i]["original_row"])
    metadata = pd.DataFrame([raw_retained[i] for i in keep_candidates])
    metadata.insert(0, "cache_index", np.arange(len(metadata), dtype=np.int64))
    metadata["class_id"] = metadata["class_id"].astype(np.int64)
    final_count = len(metadata)
    del raw_maps, raw_labels, raw_lots, df, raw_retained, resized_groups
    gc.collect()
    print(f"Resized-map cleaning retained {final_count:,}; writing uint8 cache", flush=True)
    image_path = processed / "images.npy"
    images = np.lib.format.open_memmap(image_path, mode="w+", dtype=np.uint8, shape=(final_count, 64, 64))
    for begin in range(0, final_count, 2048):
        chosen = keep_candidates[begin:begin + 2048]
        images[begin:begin + len(chosen)] = candidates[chosen]
    images.flush()
    del images, candidates, keep_candidates
    gc.collect()
    candidate_path.unlink()
    labels = metadata["class_id"].to_numpy(dtype=np.int64)
    np.save(processed / "labels.npy", labels, allow_pickle=False)
    groups = metadata["lot"].to_numpy()
    print(f"Assigning 20 stratified lot folds (seed {args.seed}); folds 0-13 train, 14-16 val, 17-19 test", flush=True)
    splitter = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=args.seed)
    folds = np.full(final_count, -1, dtype=np.int16)
    for fold, (_, indices) in enumerate(splitter.split(np.zeros(final_count, dtype=np.uint8), labels, groups)):
        folds[indices] = fold
    assert np.all(folds >= 0)
    split_indices = {"train": np.flatnonzero(folds < 14),
                     "val": np.flatnonzero((folds >= 14) & (folds < 17)),
                     "test": np.flatnonzero(folds >= 17)}
    metadata["fold"] = folds
    metadata["split"] = np.where(folds < 14, "train", np.where(folds < 17, "val", "test"))
    overlap_audit = {}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        check = {}
        for column in ["lot", "raw_hash", "resized_hash"]:
            left = set(metadata.iloc[split_indices[a]][column])
            right = set(metadata.iloc[split_indices[b]][column])
            check[column] = len(left.intersection(right))
            assert check[column] == 0, (a, b, column)
        overlap_audit[f"{a}_vs_{b}"] = check
    concatenated = np.concatenate(list(split_indices.values()))
    assert len(concatenated) == final_count and len(np.unique(concatenated)) == final_count
    assert sum(counts.values()) + final_count == total
    split_counts = {name: np.bincount(labels[idx], minlength=9) for name, idx in split_indices.items()}
    for name, class_counts in split_counts.items():
        assert np.all(class_counts > 0), f"Missing class in {name}: {class_counts}"
    assert metadata["lot"].groupby(metadata["lot"]).size().sum() == final_count
    np.savez_compressed(processed / "splits.npz", **split_indices)
    metadata.to_csv(processed / "metadata.csv", index=False, encoding="utf-8-sig")
    class_table = pd.DataFrame({"class_id": range(9), "class": CLASSES,
                                "train": split_counts["train"], "val": split_counts["val"],
                                "test": split_counts["test"], "total": np.bincount(labels, minlength=9)})
    class_table.to_csv(output / "split_class_counts.csv", index=False, encoding="utf-8-sig")
    lot_table = metadata.groupby(["split", "lot"]).agg(maps=("cache_index", "size")).reset_index()
    lot_table.to_csv(output / "split_lot_assignments.csv", index=False, encoding="utf-8-sig")
    config = {"classes": CLASSES, "class_to_id": {name: i for i, name in enumerate(CLASSES)},
              "image_shape": [64, 64], "image_dtype": "uint8", "stored_pixel_values": [0, 1, 2],
              "model_input_encoding": "torch.float32(stored_uint8) / 2.0",
              "model_input_values": [0.0, 0.5, 1.0], "wafer_mask": "stored_uint8 > 0; fixed from clean input",
              "resize": "Pillow Image.Resampling.NEAREST to width=64,height=64; no geometric augmentation",
              "seed": args.seed, "split_method": f"StratifiedGroupKFold(n_splits=20, shuffle=True, random_state={args.seed})",
              "folds": {"train": list(range(14)), "val": [14, 15, 16], "test": [17, 18, 19]}}
    save_json(processed / "config.json", config)
    paths_to_hash = [args.input.resolve(), image_path, processed / "labels.npy", processed / "splits.npz", processed / "metadata.csv", processed / "config.json"]
    print("Computing checksums and finalizing provenance", flush=True)
    file_integrity = {relative_path(p): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in paths_to_hash}
    clean_summary = {
        "status": "complete", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": relative_path(args.input), "input_sha256": file_integrity[relative_path(args.input)]["sha256"],
        "total_input_records": total, "initial_recognized_labeled": sum(initial_class_counts.values()),
        "initial_class_counts": {name: initial_class_counts[name] for name in CLASSES},
        "valid_labeled_with_lot_before_deduplication": valid_labeled,
        "original_hash_conflict_groups": raw_conflict_groups,
        "original_hash_same_label_duplicate_groups": raw_duplicate_groups,
        "after_original_map_cleaning": valid_labeled - counts["raw_label_conflict"] - counts["raw_same_label_duplicate"],
        "resized_hash_conflict_groups": resized_conflict_groups,
        "resized_hash_same_label_duplicate_groups": resized_duplicate_groups,
        "exclusions_by_reason": dict(counts), "final_retained": final_count,
        "final_class_counts": {name: int(np.count_nonzero(labels == i)) for i, name in enumerate(CLASSES)},
        "policies": [
            "Only recognized labeled maps with a lot and valid 2D values in {0,1,2} are eligible.",
            "At each identity stage, remove every map in any group with conflicting class labels; no majority voting or guessed correction.",
            "At each identity stage, retain the lowest original row position from same-label duplicates.",
            "Original identity includes shape and uint8 pixels. Resized identity is rechecked after deterministic nearest-neighbor 64x64 preprocessing.",
            "Cleaning precedes group splitting; the original trainTestLabel/trianTestLabel field is not used as the new split.",
            "Removed maps remain unchanged in the raw archive; exclusions are logged with stage and representative, where applicable.",
        ],
        "hash_identity_method": "BLAKE2b-128 over little-endian uint32 height,width and C-order uint8 values",
        "exclusions_csv": relative_path(exclusion_path),
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "Pillow": PIL.__version__, "scikit_learn": sklearn.__version__},
    }
    split_summary = {
        "status": "complete", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed, "target_ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
        "method": config["split_method"], "fold_assignment": config["folds"],
        "selection_rule": "Fixed fold assignment specified before any model training; no search or choice based on model performance.",
        "total_cleaned_maps": final_count, "total_lots": int(metadata["lot"].nunique()),
        "splits": {name: {"maps": len(idx), "fraction": len(idx) / final_count,
                           "lots": int(metadata.iloc[idx]["lot"].nunique()),
                           "class_counts": {label: int(split_counts[name][i]) for i, label in enumerate(CLASSES)}} for name, idx in split_indices.items()},
        "cross_split_intersections": overlap_audit,
        "assertions_passed": {"every_record_assigned_once": True, "nine_classes_present_in_each_split": True,
                              "lot_disjoint": True, "original_map_disjoint": True, "resized_map_disjoint": True,
                              "input_records_equal_retained_plus_excluded": True},
        "file_integrity": file_integrity, "runtime_seconds": round(time.perf_counter() - start, 3),
        "notes": ["Class weights must be computed from train labels only.",
                  "Use validation for checkpoint selection; freeze all choices before inspecting test metrics.",
                  "No oversampling, geometric augmentation, model fitting, or model-based split selection was performed."]
    }
    save_json(output / "cleaning_summary.json", clean_summary)
    save_json(output / "split_summary.json", split_summary)
    print(json.dumps({"cleaned": final_count, "exclusions": dict(counts), "splits": split_summary["splits"],
                      "intersections": overlap_audit, "seconds": split_summary["runtime_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
