import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat
from scipy.ndimage import maximum_filter


def parse_args():
    parser = argparse.ArgumentParser("Create a spatially disjoint PaviaU train/test split")
    parser.add_argument("--gt_file", required=True, type=Path)
    parser.add_argument("--gt_key", default="paviaU_gt")
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--train_ratio", default=0.10, type=float)
    parser.add_argument("--patch_size", default=15, type=int)
    parser.add_argument("--split_seed", default=2027, type=int)
    return parser.parse_args()


def spatial_scores(coords, class_id, seed):
    y = coords[:, 0].astype(np.float64)
    x = coords[:, 1].astype(np.float64)
    y = (y - y.min()) / max(float(y.max() - y.min()), 1.0)
    x = (x - x.min()) / max(float(x.max() - x.min()), 1.0)
    directions = (
        x,
        -x,
        y,
        -y,
        x + y,
        x - y,
        -x + y,
        -x - y,
    )
    return directions[(int(class_id) + int(seed)) % len(directions)]


def main():
    args = parse_args()
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train_ratio must be between 0 and 1.")
    if args.patch_size <= 0 or args.patch_size % 2 == 0:
        raise ValueError("--patch_size must be a positive odd integer.")

    data = loadmat(args.gt_file)
    if args.gt_key not in data:
        available = sorted(key for key in data if not key.startswith("__"))
        raise KeyError(f"{args.gt_key} not found. Available keys: {available}")
    labels = np.asarray(data[args.gt_key], dtype=np.int64)
    train_labels = np.zeros_like(labels)

    class_ids = [int(value) for value in np.unique(labels) if int(value) > 0]
    for class_id in class_ids:
        coords = np.argwhere(labels == class_id)
        count = max(1, int(round(len(coords) * args.train_ratio)))
        scores = spatial_scores(coords, class_id, args.split_seed)
        selected = np.argsort(scores, kind="stable")[:count]
        train_coords = coords[selected]
        train_labels[train_coords[:, 0], train_coords[:, 1]] = class_id

    # Two radius-r patches overlap when their centers are at Chebyshev distance
    # at most 2r = patch_size - 1. Remove those test centers explicitly.
    guard_radius = args.patch_size - 1
    train_guard = maximum_filter(
        (train_labels > 0).astype(np.uint8),
        size=2 * guard_radius + 1,
        mode="constant",
    ).astype(bool)
    test_labels = labels.copy()
    test_labels[train_guard] = 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "PaviaU_train_spatial.mat"
    test_path = args.output_dir / "PaviaU_test_spatial.mat"
    savemat(train_path, {args.gt_key: train_labels})
    savemat(test_path, {args.gt_key: test_labels})

    summary = {
        "source": str(args.gt_file.resolve()),
        "shape": list(labels.shape),
        "train_ratio_requested": args.train_ratio,
        "patch_size": args.patch_size,
        "guard_radius_pixels": guard_radius,
        "split_seed": args.split_seed,
        "classes": {},
    }
    for class_id in class_ids:
        total = int(np.sum(labels == class_id))
        train = int(np.sum(train_labels == class_id))
        test = int(np.sum(test_labels == class_id))
        summary["classes"][str(class_id)] = {
            "total": total,
            "train": train,
            "test": test,
            "excluded_by_guard": total - train - test,
        }

    summary_path = args.output_dir / "PaviaU_spatial_split.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved: {train_path}")
    print(f"Saved: {test_path}")


if __name__ == "__main__":
    main()
