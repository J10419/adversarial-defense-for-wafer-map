"""Audit WM-811K and export observed statistics plus real, reproducible examples.

Run from any directory; defaults are relative to this project, not the shell cwd.
This script does not split data, resize maps, train a model, or infer predictions.
Only load a pickle whose source you trust; pickle can execute Python on loading.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


from artifact_paths import relative_path

ROOT = Path(__file__).resolve().parents[1]
CLASSES = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Random", "Scratch",
    "Near-full", "None",
]
CANONICAL = {name.casefold(): name for name in CLASSES}
COLORS = ["#e8edf2", "#447b9e", "#d64b38"]


def unwrap(value):
    """Unwrap WM-811K's nested singleton arrays without conflating none and null."""
    while isinstance(value, (np.ndarray, list, tuple)):
        if len(value) == 0:
            return None
        if len(value) != 1:
            return repr(value)
        value = value[0]
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = str(value).strip()
    return text if text else None


def native(value):
    if isinstance(value, np.generic):
        return value.item()
    return value if isinstance(value, (str, int, float, bool, type(None))) else str(value)


def write_csv(rows, path, columns=None):
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False, encoding="utf-8-sig")


def draw_examples(df, samples, output, seed, counts):
    cmap = ListedColormap(COLORS)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    fig, axes = plt.subplots(3, 3, figsize=(11, 11), facecolor="white")
    manifest = []
    for ax, name in zip(axes.flat, CLASSES):
        pos = samples.get(name)
        if pos is None:
            ax.text(0.5, 0.5, "No eligible map", ha="center", va="center")
            ax.set_title(name, fontsize=14, fontweight="bold")
            ax.set_axis_off()
            continue
        row = df.iloc[pos]
        arr = np.asarray(row["waferMap"])
        valid_dies = int(np.count_nonzero(arr > 0))
        failed_dies = int(np.count_nonzero(arr == 2))
        ax.imshow(arr, cmap=cmap, norm=norm, interpolation="nearest", aspect="equal")
        ax.set_title(f"{name}\n{arr.shape[0]} x {arr.shape[1]} original map", fontsize=13, pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        manifest.append({
            "class": name,
            "row_position_zero_based": pos,
            "original_dataframe_index": native(df.index[pos]),
            "lot": unwrap(row.get("lotName")),
            "wafer_index": unwrap(row.get("waferIndex")),
            "original_failure_type": unwrap(row["failureType"]),
            "height": arr.shape[0], "width": arr.shape[1],
            "valid_dies": valid_dies, "failing_dies": failed_dies,
            "failing_die_fraction": failed_dies / valid_dies,
            "seed": seed,
            "eligible_maps_in_class": counts[name],
        })
    fig.suptitle("WM-811K | Nine labeled wafer-map classes", fontsize=19, fontweight="bold", y=0.98)
    fig.legend(
        handles=[Patch(facecolor=c, label=s) for c, s in zip(COLORS, ["0  Background / no die", "1  Passing die", "2  Failing die"])],
        loc="lower center", bbox_to_anchor=(0.5, 0.064), ncol=3, frameon=False, fontsize=11,
    )
    fig.text(0.5, 0.026, f"Seed {seed} | Randomly selected real maps, not model predictions.\nOriginal shapes and discrete values are preserved. None means no labeled systematic pattern.",
             ha="center", va="center", fontsize=10, color="#485360")
    fig.subplots_adjust(top=0.91, bottom=0.13, left=0.05, right=0.95, hspace=0.33, wspace=0.25)
    fig.savefig(output / "nine_class_examples.png", dpi=240, facecolor="white")
    plt.close(fig)
    write_csv(manifest, output / "example_manifest.csv")


def draw_distribution(counts, labeled_total, output):
    fig, ax = plt.subplots(figsize=(10.5, 6.5), facecolor="white")
    values = [counts[n] for n in CLASSES]
    colors = ["#386f94"] * 8 + ["#bf7044"]
    bars = ax.barh(CLASSES, values, color=colors, height=0.65)
    ax.invert_yaxis()
    ax.set_xscale("log")
    maximum = max(values, default=1)
    # Leave space for the longest numeric label, including the dominant None class.
    ax.set_xlim(0.8, max(10, maximum * 25))
    ax.set_xlabel("Number of maps (logarithmic scale)", fontsize=11)
    ax.set_title("WM-811K labeled class distribution", fontsize=17, fontweight="bold", loc="left", pad=24)
    for bar, n in zip(bars, values):
        pct = 100 * n / labeled_total if labeled_total else 0
        ax.text(max(1, n) * 1.08, bar.get_y() + bar.get_height() / 2,
                f"{n:,}  ({pct:.2f}%)", va="center", fontsize=10)
    ax.set_axisbelow(True)
    ax.grid(axis="x", which="major", alpha=0.22)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=11)
    fig.text(0.13, 0.035, f"Percentages use the {labeled_total:,} maps with one of the nine recognized labels.\nUnlabeled records are excluded; None is a labeled class.", fontsize=10, color="#485360")
    fig.subplots_adjust(left=0.13, right=0.96, top=0.86, bottom=0.16)
    fig.savefig(output / "class_distribution.png", dpi=240, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "raw" / "LSWMD.pkl")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "data")
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--skip-duplicates", action="store_true", help="Skip exact-duplicate checks on recognized labeled maps.")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    print(f"Loading {args.input.resolve()} ...", flush=True)
    df = pd.read_pickle(args.input)
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame in the WM-811K pickle.")
    for required in ["waferMap", "failureType"]:
        if required not in df.columns:
            raise KeyError(f"Missing required column: {required}")
    print(f"Loaded {len(df):,} records with columns {list(df.columns)}", flush=True)

    counts, statuses, original_labels, unexpected_labels = Counter(), Counter(), Counter(), Counter()
    shapes, shape_by_status, pixels = Counter(), Counter(), Counter()
    original_split, original_split_by_class = Counter(), Counter()
    lots_all, lots_labeled = set(), set()
    lots_per_class = defaultdict(set)
    missing_lot = 0
    issues, issue_examples = Counter(), defaultdict(list)
    eligible_counts, samples = Counter(), {}
    # Reservoir sampling uses one random generator per class, independent of interleaving.
    rngs = {name: np.random.default_rng(s) for name, s in zip(CLASSES, np.random.SeedSequence(args.seed).spawn(9))}
    hashes = {}
    duplicate_eligible_count = 0
    class_index = {name: i for i, name in enumerate(CLASSES)}
    label_values = df["failureType"].array
    lot_values = df["lotName"].array if "lotName" in df else None
    # The original Kaggle pickle uses the misspelling 'trianTestLabel'.
    split_column = next((c for c in ["trainTestLabel", "trianTestLabel"] if c in df), None)
    split_values = df[split_column].array if split_column is not None else None

    def issue(name, pos):
        issues[name] += 1
        if len(issue_examples[name]) < 20:
            issue_examples[name].append(pos)

    for pos, raw in enumerate(df["waferMap"].array):
        original = unwrap(label_values[pos])
        original_labels[original if original is not None else "<missing>"] += 1
        label = CANONICAL.get(original.casefold()) if original is not None else None
        if original is None:
            status = "unlabeled"
        elif label is None:
            status = "unexpected_label"
            unexpected_labels[original] += 1
        else:
            status = "recognized_labeled"
            counts[label] += 1
        statuses[status] += 1
        lot = unwrap(lot_values[pos]) if lot_values is not None else None
        if lot is None:
            missing_lot += 1
        else:
            lots_all.add(lot)
            if label is not None:
                lots_labeled.add(lot)
                lots_per_class[label].add(lot)
        split = unwrap(split_values[pos]) if split_values is not None else None
        split = split if split is not None else "<missing>"
        original_split[split] += 1
        original_split_by_class[(label if label is not None else status, split)] += 1

        arr = np.asarray(raw)
        if arr.ndim != 2:
            issue("non_2d_map", pos)
            continue
        shape = (int(arr.shape[0]), int(arr.shape[1]))
        shapes[shape] += 1
        shape_by_status[(status, *shape)] += 1
        if not arr.size:
            issue("empty_map", pos)
            continue
        # Process one variable-size map at a time; never stack the entire data set.
        values, frequencies = np.unique(arr, return_counts=True)
        for value, count in zip(values, frequencies):
            pixels[str(native(value))] += int(count)
        values_valid = bool(np.isin(values, [0, 1, 2]).all())
        if not values_valid:
            issue("unexpected_pixel_values", pos)
            continue
        valid_dies = int(np.count_nonzero(arr > 0))
        if valid_dies == 0:
            issue("no_valid_die", pos)
            continue
        if not np.any(arr == 2):
            issue("zero_failing_die_map", pos)
        if label is not None:
            eligible_counts[label] += 1
            if rngs[label].integers(eligible_counts[label]) == 0:
                samples[label] = pos
            if not args.skip_duplicates:
                duplicate_eligible_count += 1
                hasher = hashlib.blake2b(digest_size=16)
                hasher.update(struct.pack("<II", *shape))
                hasher.update(np.asarray(arr, dtype=np.uint8).tobytes(order="C"))
                digest = hasher.digest()
                bit = 1 << class_index[label]
                if digest in hashes:
                    hashes[digest][0] += 1
                    hashes[digest][2] |= bit
                else:
                    hashes[digest] = [1, pos, bit]
        if (pos + 1) % 100_000 == 0:
            print(f"Audited {pos + 1:,}/{len(df):,} records ({time.perf_counter()-start:.1f}s)", flush=True)

    total = len(df)
    labeled = sum(counts.values())
    assert sum(statuses.values()) == total
    assert sum(original_labels.values()) == total
    assert sum(original_split.values()) == total
    assert labeled == statuses["recognized_labeled"]
    class_rows = [{"class": name, "count": counts[name],
                   "pct_labeled": 100 * counts[name] / labeled if labeled else 0,
                   "pct_all": 100 * counts[name] / total if total else 0,
                   "unique_lots": len(lots_per_class[name]),
                   "eligible_example_maps": eligible_counts[name]} for name in CLASSES]
    write_csv(class_rows, output / "class_counts.csv")
    write_csv([{"label_status": k, "count": statuses[k], "pct_all": 100 * statuses[k] / total if total else 0}
               for k in ["recognized_labeled", "unlabeled", "unexpected_label"]], output / "label_status_counts.csv")
    write_csv([{"original_label": k, "count": v} for k, v in sorted(original_labels.items())], output / "original_label_counts.csv")
    write_csv([{"height": h, "width": w, "count": n} for (h, w), n in shapes.most_common()], output / "map_shape_counts.csv")
    write_csv([{"label_status": s, "height": h, "width": w, "count": n} for (s, h, w), n in sorted(shape_by_status.items())], output / "map_shape_counts_by_status.csv")
    write_csv([{"original_train_test_label": k, "count": v} for k, v in sorted(original_split.items())], output / "original_split_counts.csv")
    write_csv([{"class_or_status": c, "original_train_test_label": s, "count": n} for (c, s), n in sorted(original_split_by_class.items())], output / "original_split_by_class.csv")
    write_csv([{"pixel_value": k, "count": v} for k, v in sorted(pixels.items())], output / "pixel_value_counts.csv")

    duplicates = []
    if not args.skip_duplicates:
        for digest, (n, first, mask) in hashes.items():
            if n > 1:
                labels = [name for name in CLASSES if mask & (1 << class_index[name])]
                duplicates.append({"blake2b_128": digest.hex(), "count": n,
                                   "first_row_position_zero_based": first,
                                   "labels": "|".join(labels), "conflicting_labels": len(labels) > 1})
        write_csv(duplicates, output / "labeled_exact_duplicate_groups.csv",
                  columns=["blake2b_128", "count", "first_row_position_zero_based", "labels", "conflicting_labels"])

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": relative_path(args.input), "input_bytes": args.input.stat().st_size,
        "record_count": total, "columns": [str(c) for c in df.columns],
        "recognized_labeled_count": labeled, "unlabeled_count": statuses["unlabeled"],
        "unexpected_label_count": statuses["unexpected_label"], "unexpected_labels": dict(unexpected_labels),
        "class_counts": {name: counts[name] for name in CLASSES},
        "class_order": CLASSES, "unique_lots_all_records": len(lots_all),
        "unique_lots_recognized_labeled": len(lots_labeled), "missing_lot_records": missing_lot,
        "unique_lots_per_class": {name: len(lots_per_class[name]) for name in CLASSES},
        "unique_map_shapes": len(shapes),
        "map_height_min_max": [min((s[0] for s in shapes), default=None), max((s[0] for s in shapes), default=None)],
        "map_width_min_max": [min((s[1] for s in shapes), default=None), max((s[1] for s in shapes), default=None)],
        "pixel_value_counts": dict(pixels), "original_train_test_label_column": split_column,
        "original_train_test_label_counts": dict(original_split),
        "map_audit_counts": dict(issues), "map_audit_example_row_positions": dict(issue_examples),
        "exact_duplicate_check": {
            "performed": not args.skip_duplicates,
            "scope": "Recognized labeled maps only; 2D, values in {0,1,2}, and at least one valid die. Original shape and values, before resizing.",
            "maps_checked": duplicate_eligible_count,
            "unique_hashes": len(hashes), "duplicate_groups": len(duplicates),
            "maps_in_duplicate_groups": sum(d["count"] for d in duplicates),
            "redundant_maps_beyond_first": sum(d["count"] - 1 for d in duplicates),
            "conflicting_label_groups": sum(d["conflicting_labels"] for d in duplicates),
            "hash_method": "BLAKE2b-128 over little-endian uint32 shape followed by row-major uint8 values; hash equality is used for grouping.",
        },
        "example_selection": {
            "seed": args.seed,
            "method": "Uniform reservoir sample per class with independent NumPy random streams; samples are illustrative real observations, not selected for typicality or prediction outcome.",
            "eligibility": "2D nonempty map; all values in {0,1,2}; at least one valid die; recognized class label.",
        },
        "notes": [
            "The original lowercase label 'none' is displayed as 'None'. Missing labels are a separate category.",
            "None denotes no labeled systematic failure pattern, not necessarily zero failing dies.",
            "zero_failing_die_map is descriptive, not automatically an invalid record.",
            "Original trainTestLabel values are reported for auditing only; no new train/validation/test split has been made.",
            "No resizing, augmentation, model training, attacks, or predictions were performed.",
        ],
        "runtime_seconds_before_figures": round(time.perf_counter() - start, 3),
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
    }
    (output / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    draw_examples(df, samples, output, args.seed, eligible_counts)
    draw_distribution(counts, labeled, output)
    print(json.dumps({"total": total, "labeled": labeled, "unlabeled": statuses["unlabeled"],
                      "classes": summary["class_counts"], "duplicate_groups": len(duplicates),
                      "output": str(output), "elapsed_seconds": round(time.perf_counter() - start, 2)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
