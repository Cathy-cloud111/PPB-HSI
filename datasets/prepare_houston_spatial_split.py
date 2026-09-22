"""Create a guarded, patch-disjoint Houston2013 train/test split.

The split is generated from the complete ground-truth map.  For each class,
training centres are taken from one spatial extreme; among eight deterministic
directions, the direction retaining the most same-class test centres after the
patch guard is selected.  Test centres inside the global guard of any training
centre are removed, so no 15x15 train and test patches share scene pixels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat
from scipy.ndimage import maximum_filter


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-file", required=True, type=Path)
    parser.add_argument("--gt-key", default="gt")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-ratio", default=0.10, type=float)
    parser.add_argument("--patch-size", default=15, type=int)
    parser.add_argument("--split-seed", default=2027, type=int)
    parser.add_argument("--min-test-per-class", default=20, type=int)
    return parser.parse_args()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def normalized_axes(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = coords[:, 0].astype(np.float64)
    x = coords[:, 1].astype(np.float64)
    y = (y - y.min()) / max(float(y.max() - y.min()), 1.0)
    x = (x - x.min()) / max(float(x.max() - x.min()), 1.0)
    return y, x


def directional_scores(coords: np.ndarray) -> tuple[np.ndarray, ...]:
    y, x = normalized_axes(coords)
    return (x, -x, y, -y, x + y, x - y, -x + y, -x - y)


def choose_training_region(labels: np.ndarray, class_id: int, count: int,
                           guard_size: int, split_seed: int):
    coords = np.argwhere(labels == class_id)
    directions = directional_scores(coords)
    start = (class_id + split_seed) % len(directions)
    candidates = []
    for offset in range(len(directions)):
        direction_id = (start + offset) % len(directions)
        order = np.argsort(directions[direction_id], kind="stable")
        selected = coords[order[:count]]
        mask = np.zeros(labels.shape, dtype=np.uint8)
        mask[selected[:, 0], selected[:, 1]] = 1
        guard = maximum_filter(mask, size=guard_size, mode="constant", cval=0).astype(bool)
        retained = int(np.sum((labels == class_id) & ~guard))
        candidates.append((retained, -offset, direction_id, selected))
    return max(candidates, key=lambda item: (item[0], item[1]))


def main():
    args = parse_args()
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be between zero and one")
    if args.patch_size <= 0 or args.patch_size % 2 == 0:
        raise ValueError("--patch-size must be a positive odd integer")
    if args.min_test_per_class < 1:
        raise ValueError("--min-test-per-class must be positive")

    source = loadmat(args.gt_file)
    if args.gt_key not in source:
        available = sorted(key for key in source if not key.startswith("__"))
        raise KeyError(f"{args.gt_key!r} not found; available keys: {available}")
    labels = np.asarray(source[args.gt_key])
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.number):
        raise ValueError("Ground truth must be one numeric 2-D label map")
    if not np.isfinite(labels).all() or (labels < 0).any() or not np.equal(labels, np.floor(labels)).all():
        raise ValueError("Ground-truth labels must be finite nonnegative integers")
    labels = labels.astype(np.int64, copy=False)
    class_ids = [int(value) for value in np.unique(labels) if int(value) > 0]
    if class_ids != list(range(1, max(class_ids) + 1)):
        raise ValueError(f"Expected contiguous class IDs, found {class_ids}")

    # Two odd-width square patches overlap when their centres are within
    # patch_size-1 pixels in both axes.  The corresponding maximum-filter
    # window has width 2*(patch_size-1)+1.
    guard_radius = args.patch_size - 1
    guard_size = 2 * guard_radius + 1
    train_labels = np.zeros_like(labels)
    selections = {}
    for class_id in class_ids:
        coords = np.argwhere(labels == class_id)
        count = max(1, int(round(len(coords) * args.train_ratio)))
        retained, _, direction_id, selected = choose_training_region(
            labels, class_id, count, guard_size, args.split_seed)
        train_labels[selected[:, 0], selected[:, 1]] = class_id
        selections[str(class_id)] = {
            "requested_train": count,
            "direction_id": int(direction_id),
            "same_class_test_after_local_guard": retained,
        }

    global_guard = maximum_filter(
        (train_labels > 0).astype(np.uint8), size=guard_size,
        mode="constant", cval=0,
    ).astype(bool)
    test_labels = labels.copy()
    test_labels[global_guard] = 0

    train_centres = train_labels > 0
    test_centres = test_labels > 0
    train_neighbourhood = maximum_filter(
        train_centres.astype(np.uint8), size=guard_size,
        mode="constant", cval=0,
    ).astype(bool)
    overlap = int(np.sum(test_centres & train_neighbourhood))
    if overlap:
        raise RuntimeError(f"Guard construction failed: {overlap} test centres overlap train patches")

    classes = {}
    for class_id in class_ids:
        total = int(np.sum(labels == class_id))
        train = int(np.sum(train_labels == class_id))
        test = int(np.sum(test_labels == class_id))
        if test < args.min_test_per_class:
            raise ValueError(
                f"Class {class_id} retains only {test} test centres; "
                "reduce --train-ratio or --patch-size"
            )
        classes[str(class_id)] = {
            "total": total,
            "train": train,
            "test": test,
            "excluded_by_guard": total - train - test,
            **selections[str(class_id)],
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "TRLabel.mat"
    test_path = args.output_dir / "TSLabel.mat"
    savemat(train_path, {"TRLabel": train_labels.astype(np.uint8)}, do_compression=True)
    savemat(test_path, {"TSLabel": test_labels.astype(np.uint8)}, do_compression=True)
    report = {
        "dataset": "Houston2013",
        "source": str(args.gt_file.resolve()),
        "source_sha256": digest(args.gt_file),
        "shape": list(labels.shape),
        "train_ratio_requested": args.train_ratio,
        "patch_size": args.patch_size,
        "guard_radius_pixels": guard_radius,
        "split_seed": args.split_seed,
        "train_test_patch_overlap_count": overlap,
        "spatially_disjoint_patches": overlap == 0,
        "classes": classes,
    }
    report_path = args.output_dir / "Houston2013_spatial_split.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"Saved: {train_path}")
    print(f"Saved: {test_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
