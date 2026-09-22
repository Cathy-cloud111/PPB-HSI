from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from scipy.io import loadmat
except ImportError:  # pragma: no cover
    loadmat = None


HOUSTON2013_CLASS_NAMES = [
    "healthy grass",
    "stressed grass",
    "synthetic grass",
    "trees",
    "soil",
    "water",
    "residential",
    "commercial",
    "road",
    "highway",
    "railway",
    "parking lot 1",
    "parking lot 2",
    "tennis court",
    "running track",
]


def parse_class_sessions(session_text):
    """Parse strings like '1-9,10-12,13-15' into 1-based class-id sessions."""
    sessions = []
    for chunk in str(session_text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, end = chunk.split("-", 1)
            sessions.append(list(range(int(start), int(end) + 1)))
        else:
            sessions.append([int(chunk)])
    if not sessions:
        raise ValueError("No class sessions were parsed.")
    return sessions


def parse_band_indices(band_indices):
    if band_indices is None or str(band_indices).strip() == "":
        return None
    return [int(idx.strip()) for idx in str(band_indices).split(",") if idx.strip()]


def load_mat_array(path, key=None, ndim=None):
    if loadmat is None:
        raise ImportError("scipy is required to read Houston .mat files.")

    path = Path(path)
    data = loadmat(path)
    if key:
        if key not in data:
            available = [name for name in data if not name.startswith("__")]
            raise KeyError(f"{key} not found in {path}. Available keys: {available}")
        return np.asarray(data[key])

    candidates = []
    for name, value in data.items():
        if name.startswith("__") or not isinstance(value, np.ndarray):
            continue
        if ndim is None or value.ndim == ndim:
            candidates.append((name, value))
    if not candidates:
        raise KeyError(f"No {ndim}D array found in {path}. Pass the key explicitly.")
    return np.asarray(candidates[0][1])


def minmax_per_band(hsi):
    hsi = hsi.astype(np.float32, copy=False)
    band_min = hsi.min(axis=(0, 1), keepdims=True)
    band_max = hsi.max(axis=(0, 1), keepdims=True)
    denom = np.maximum(band_max - band_min, 1e-6)
    return (hsi - band_min) / denom


def load_houston_hsi(hsi_file, hsi_key=None, band_indices="", normalize_per_band=True):
    hsi = load_mat_array(hsi_file, hsi_key, ndim=3).astype(np.float32)
    bands = parse_band_indices(band_indices)
    if bands is not None:
        hsi = hsi[:, :, bands]
    if normalize_per_band:
        hsi = minmax_per_band(hsi)
    return np.ascontiguousarray(hsi)


def load_houston_lidar(lidar_file, lidar_key=None, normalize_per_band=True):
    lidar = load_mat_array(lidar_file, lidar_key, ndim=None).astype(np.float32)
    lidar = np.squeeze(lidar)
    if lidar.ndim == 2:
        lidar = lidar[:, :, None]
    elif lidar.ndim == 3:
        if lidar.shape[0] <= 8 and lidar.shape[-1] > 8:
            lidar = np.moveaxis(lidar, 0, -1)
    else:
        raise ValueError(f"LiDAR array must be 2D or 3D, got shape {lidar.shape}.")
    if normalize_per_band:
        lidar = minmax_per_band(lidar)
    return np.ascontiguousarray(lidar)


def fuse_hsi_lidar(hsi, lidar):
    if hsi.shape[:2] != lidar.shape[:2]:
        raise ValueError(f"HSI shape {hsi.shape[:2]} does not match LiDAR shape {lidar.shape[:2]}.")
    return np.ascontiguousarray(np.concatenate([hsi, lidar], axis=2))


class HoustonPatchClassification(Dataset):
    """Pixel-centered patch classification dataset for Houston2013.

    Label maps use 1-based class ids and 0 for unlabeled/background. Returned
    labels are converted to 0-based ids so they can be used with cross entropy.
    """

    def __init__(
        self,
        hsi_file,
        label_file,
        class_ids,
        hsi_key=None,
        label_key=None,
        band_indices="",
        patch_size=15,
        normalize_per_band=True,
        max_samples_per_class=0,
        max_samples_by_class=None,
        sample_filter=None,
        seed=42,
        hsi_array=None,
    ):
        self.patch_size = int(patch_size)
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive.")
        self.class_ids = sorted(int(cls_id) for cls_id in class_ids)

        self.hsi = hsi_array
        if self.hsi is None:
            self.hsi = load_houston_hsi(
                hsi_file,
                hsi_key=hsi_key,
                band_indices=band_indices,
                normalize_per_band=normalize_per_band,
            )
        self.labels = load_mat_array(label_file, label_key, ndim=2).astype(np.int64)
        if self.hsi.shape[:2] != self.labels.shape:
            raise ValueError(f"HSI shape {self.hsi.shape[:2]} does not match label shape {self.labels.shape}.")

        self.samples = self._build_samples(
            max_samples_per_class=max_samples_per_class,
            max_samples_by_class=max_samples_by_class,
            sample_filter=sample_filter,
            seed=seed,
        )
        if not self.samples:
            raise ValueError(f"No labeled samples found for classes {self.class_ids} in {label_file}.")

        half = self.patch_size // 2
        self.padded_hsi = np.pad(self.hsi, ((half, half), (half, half), (0, 0)), mode="edge")

    @property
    def num_channels(self):
        return int(self.hsi.shape[2])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        y, x, class_id = self.samples[idx]
        half = self.patch_size // 2
        y0 = y
        x0 = x
        patch = self.padded_hsi[y0:y0 + self.patch_size, x0:x0 + self.patch_size, :]
        patch = np.ascontiguousarray(np.moveaxis(patch, 2, 0))
        return torch.from_numpy(patch).float(), torch.tensor(class_id - 1, dtype=torch.long)

    def class_counts(self):
        counts = {class_id: 0 for class_id in self.class_ids}
        for _, _, class_id in self.samples:
            counts[class_id] = counts.get(class_id, 0) + 1
        return counts

    def _build_samples(self, max_samples_per_class=0, max_samples_by_class=None, sample_filter=None, seed=42):
        rng = np.random.default_rng(seed)
        samples = []
        max_samples_per_class = int(max_samples_per_class)
        max_samples_by_class = {
            int(class_id): int(limit)
            for class_id, limit in (max_samples_by_class or {}).items()
        }

        for class_id in self.class_ids:
            coords = np.argwhere(self.labels == class_id)
            if coords.size == 0:
                continue
            class_limit = max_samples_by_class.get(class_id, max_samples_per_class)
            if class_limit > 0 and len(coords) > class_limit:
                if sample_filter is not None:
                    selected = sample_filter(class_id, coords, class_limit)
                    if selected is not None:
                        coords = np.asarray(selected, dtype=np.int64)
                    else:
                        chosen = rng.choice(len(coords), size=class_limit, replace=False)
                        coords = coords[chosen]
                else:
                    chosen = rng.choice(len(coords), size=class_limit, replace=False)
                    coords = coords[chosen]
            for y, x in coords:
                samples.append((int(y), int(x), int(class_id)))

        rng.shuffle(samples)
        return samples


def confusion_to_metrics(confusion, class_ids):
    """Compute OA, AA and Kappa from a full confusion matrix.

    class_ids are 1-based ground-truth ids to include in the metric. Predictions
    outside this subset are still counted as errors, which is important when
    measuring current/base classes after incremental sessions.
    """
    indices = [int(class_id) - 1 for class_id in class_ids]
    rows = confusion[indices, :].astype(np.float64)
    total = float(rows.sum())
    if total <= 0:
        return {
            "oa": 0.0,
            "aa": 0.0,
            "kappa": 0.0,
            "total": 0,
            "correct": 0,
            "per_class_acc": {},
        }

    correct = float(sum(confusion[idx, idx] for idx in indices))
    oa = correct / total
    row_sum = rows.sum(axis=1)
    col_sum = confusion[:, indices].astype(np.float64).sum(axis=0)
    valid = row_sum > 0
    per_class = np.zeros(len(indices), dtype=np.float64)
    diag = np.array([confusion[idx, idx] for idx in indices], dtype=np.float64)
    per_class[valid] = diag[valid] / row_sum[valid]
    aa = float(per_class[valid].mean()) if valid.any() else 0.0
    pe = float((row_sum * col_sum).sum() / (total * total))
    kappa = float((oa - pe) / (1.0 - pe)) if abs(1.0 - pe) > 1e-12 else 0.0

    per_class_acc = {}
    for idx, class_id in enumerate(class_ids):
        if row_sum[idx] > 0:
            per_class_acc[int(class_id)] = float(per_class[idx])

    return {
        "oa": float(oa),
        "aa": float(aa),
        "kappa": float(kappa),
        "total": int(total),
        "correct": int(correct),
        "per_class_acc": per_class_acc,
    }
