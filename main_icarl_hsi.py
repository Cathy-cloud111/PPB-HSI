"""iCaRL HSI adaptation: BCE targets, persistent images, herding and NME.

Independent entry point; does not patch or modify the existing experiment engine.
This is not a reproduction of the original CIFAR/ImageNet numerical results.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from datasets.houston_cls import (
    HoustonPatchClassification, confusion_to_metrics, load_houston_hsi,
    load_mat_array, parse_class_sessions,
)
from main_houston_cls import HoustonPatchCNN, seed_everything


def get_args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("hsi_file", "train_label_file", "test_label_file", "output_dir"):
        p.add_argument("--" + flag, required=True)
    p.add_argument("--hsi_key", default="HSI")
    p.add_argument("--hsi_label_key", default=None)
    p.add_argument("--sessions", default="1-9,10-12,13-15")
    p.add_argument("--num_classes", type=int, default=15)
    p.add_argument("--hsi_patch_size", type=int, default=15)
    p.add_argument("--hsi_normalize_per_band", type=int, choices=(0, 1), default=1)
    p.add_argument("--memory_budget", type=int, required=True)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=.001)
    p.add_argument("--weight_decay", type=float, default=.0001)
    p.add_argument("--dropout", type=float, default=.2)
    p.add_argument("--max_train_samples_per_class", type=int, default=0)
    p.add_argument("--max_test_samples_per_class", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--allow-cpu", action="store_true")
    return p


def validate_args(args):
    sessions = parse_class_sessions(args.sessions)
    flat = [c for session in sessions for c in session]
    if len(flat) != len(set(flat)) or sorted(flat) != list(range(1, args.num_classes + 1)):
        raise ValueError("Sessions must partition all class IDs without overlap")
    if args.memory_budget < args.num_classes:
        raise ValueError("Memory must permit at least one exemplar per final class")
    if min(args.epochs, args.batch_size, args.threads) < 1 or args.lr <= 0:
        raise ValueError("Positive epochs, batch size, threads and LR required")
    if args.hsi_patch_size < 5 or args.hsi_patch_size % 2 != 1:
        raise ValueError("CNN requires an odd patch size >=5")
    if args.smoke and not (args.epochs == 1 and
                          0 < args.max_train_samples_per_class <= 8 and
                          0 < args.max_test_samples_per_class <= 12):
        raise ValueError("Smoke must use one epoch and capped train/test samples")
    if not args.smoke and (args.max_train_samples_per_class or args.max_test_samples_per_class):
        raise ValueError("Formal comparison must use the complete supplied split")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
    if args.device == "cpu" and not args.smoke and not args.allow_cpu:
        raise RuntimeError("Formal CPU training refused; use the server")
    return sessions


class MemoryDataset(Dataset):
    """Reads stored image patches only, never reselects from old label maps."""
    def __init__(self, memory):
        self.entries = [(entry["images"], i, c)
                        for c, entry in sorted(memory.items())
                        for i in range(len(entry["images"]))]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        images, i, c = self.entries[idx]
        return images[i], torch.tensor(c - 1, dtype=torch.long)


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        image, label = self.dataset[i]
        return image, label, i


def loader(dataset, args, shuffle=False, session=0):
    # Dedicated generators isolate order from diagnostics and test iterators.
    generator = torch.Generator().manual_seed(args.seed + 1009 * session)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=args.device == "cuda",
                      generator=generator, drop_last=False)


def icarl_targets(labels, seen, old, old_probabilities=None):
    """New outputs use indicators; ALL old outputs use cached teacher sigmoids."""
    seen = list(seen)
    if len(seen) != len(set(seen)) or not set(old).issubset(seen):
        raise ValueError("Invalid seen/old classes")
    if not set((labels + 1).tolist()).issubset(seen):
        raise ValueError("Target label outside seen classes")
    indices = torch.tensor([c - 1 for c in seen], device=labels.device)
    target = (labels[:, None] == indices[None, :]).float()
    if old:
        if old_probabilities is None or old_probabilities.shape != (len(labels), len(old)):
            raise ValueError("Every example needs the complete old-output target")
        target[:, [seen.index(c) for c in old]] = old_probabilities.detach()
    return target


def normalized_features(features):
    if not torch.isfinite(features).all() or (features.norm(dim=1) <= 1e-12).any():
        raise ValueError("Nonfinite or zero features cannot define spherical NME")
    return F.normalize(features, dim=1)


def herding_indices(features, count):
    """Ordered no-replacement approximation to the normalized class mean.

    All candidate vectors and candidate averages are re-normalized, following
    the paper's spherical convention. Stable input order breaks exact ties.
    """
    features = normalized_features(features.detach().cpu().double())
    if count < 1 or len(features) < 1:
        raise ValueError("Herding needs positive count and candidates")
    target = normalized_features(features.mean(0, keepdim=True))[0]
    selected, total = [], torch.zeros(features.shape[1], dtype=features.dtype)
    available = torch.ones(len(features), dtype=torch.bool)
    for _ in range(min(count, len(features))):
        candidate_means = F.normalize(features + total, dim=1)
        distances = (candidate_means - target).square().sum(1)
        distances[~available] = float("inf")
        choice = int(distances.argmin())
        selected.append(choice)
        available[choice] = False
        total += features[choice]
    return selected


def reduce_memory(memory, count):
    if count < 1:
        raise ValueError("Cannot discard every exemplar of a seen class")
    # Keep prioritized prefixes. No old-class re-herding or full-data access.
    return {c: {"images": entry["images"][:count].clone(),
                "coordinates": entry["coordinates"][:count]}
            for c, entry in memory.items()}


@torch.no_grad()
def extract(model, dataset, args):
    model.eval()
    return torch.cat([normalized_features(model.extract_features(x.to(args.device))).cpu()
                      for x, _ in loader(dataset, args)], dim=0)


@torch.no_grad()
def cache_old_targets(model, dataset, old, args):
    model.eval()
    indices = torch.tensor([c - 1 for c in old], device=args.device)
    return torch.cat([model(x.to(args.device)).index_select(1, indices).sigmoid().cpu()
                      for x, _ in loader(dataset, args)], dim=0)


def exemplar_means(model, memory, args):
    means = []
    for c in sorted(memory):
        features = extract(model, MemoryDataset({c: memory[c]}), args)
        means.append(normalized_features(features.mean(0, keepdim=True))[0])
    return torch.stack(means)


def nme_predict(features, means, class_ids):
    features, means = normalized_features(features), normalized_features(means)
    if len(means) != len(class_ids):
        raise ValueError("Mean/class mismatch")
    ids = torch.tensor([c - 1 for c in class_ids], device=features.device)
    return ids[(features @ means.to(features.device).T).argmax(1)]


@torch.no_grad()
def evaluate_nme(model, dataset, means, class_ids, args):
    model.eval()
    confusion = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
    for x, labels in loader(dataset, args):
        predictions = nme_predict(model.extract_features(x.to(args.device)), means, class_ids).cpu().numpy()
        np.add.at(confusion, (labels.numpy(), predictions), 1)
    return confusion


def metrics_row(confusion, seen, current, base, base_initial):
    metrics = confusion_to_metrics(confusion, seen)
    base_oa = confusion_to_metrics(confusion, base)["oa"]
    if base_initial is None:
        base_initial = base_oa
    return {**{k: metrics[k] for k in ("oa", "aa", "kappa")},
            "current_oa": confusion_to_metrics(confusion, current)["oa"],
            "base_oa": base_oa, "apd_base_oa_raw": base_initial - base_oa,
            "apd_base_forgetting": max(0., base_initial - base_oa)}, base_initial


def main(args):
    sessions = validate_args(args)
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # A queue may place its manifest/log here, but training artifacts are immutable.
    if list(output.glob("session_*.pth")) or (output / "metrics.json").exists():
        raise FileExistsError("Existing training artifacts; use a fresh output directory")
    train_map = load_mat_array(args.train_label_file, args.hsi_label_key, ndim=2)
    test_map = load_mat_array(args.test_label_file, args.hsi_label_key, ndim=2)
    if train_map.shape != test_map.shape or ((train_map > 0) & (test_map > 0)).any():
        raise ValueError("Train/test center maps overlap or have different shapes")
    hsi = load_houston_hsi(args.hsi_file, args.hsi_key,
                           normalize_per_band=bool(args.hsi_normalize_per_band))
    model = HoustonPatchCNN(hsi.shape[2], args.num_classes, args.dropout).to(args.device)
    memory, seen, rows, base_initial = {}, [], [], None
    dataset_args = dict(hsi_file=args.hsi_file, hsi_array=hsi, hsi_key=args.hsi_key,
                        label_key=args.hsi_label_key, patch_size=args.hsi_patch_size,
                        seed=args.seed, normalize_per_band=bool(args.hsi_normalize_per_band))
    for session, current in enumerate(sessions, 1):
        old = list(seen)
        seen += current
        # Select CURRENT data only. Old samples come exclusively from images in memory.
        current_data = HoustonPatchClassification(label_file=args.train_label_file,
            class_ids=current, max_samples_per_class=args.max_train_samples_per_class, **dataset_args)
        old_memory_count = sum(len(entry["images"]) for entry in memory.values())
        train_data = ConcatDataset([current_data, MemoryDataset(memory)]) if memory else current_data
        old_targets = cache_old_targets(model, train_data, old, args) if old else None
        seen_indices = torch.tensor([c - 1 for c in seen], device=args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        train_loader = loader(IndexedDataset(train_data), args, shuffle=True, session=session)
        print(f"Session {session}: seen={seen}; current={len(current_data)}; replay={old_memory_count}", flush=True)
        for epoch in range(1, args.epochs + 1):
            model.train()  # iCaRL representation remains trainable after the base session.
            total, loss_sum = 0, 0.
            for x, labels, sample_indices in train_loader:
                x, labels = x.to(args.device), labels.to(args.device)
                probabilities = old_targets[sample_indices].to(args.device) if old else None
                target = icarl_targets(labels, seen, old, probabilities)
                logits = model(x).index_select(1, seen_indices)
                # Sum output BCE, average examples: original objective up to reduction.
                loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none").sum(1).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += len(labels)
                loss_sum += float(loss.detach()) * len(labels)
            if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                print(f"  epoch {epoch:03}: BCE={loss_sum / total:.6f}", flush=True)
        quota = args.memory_budget // len(seen)
        memory = reduce_memory(memory, quota)
        features = extract(model, current_data, args)
        for c in current:
            indices = [i for i, (_, _, cls) in enumerate(current_data.samples) if cls == c]
            if not indices:
                raise ValueError(f"Class {c} lacks current training samples")
            chosen = [indices[i] for i in herding_indices(features[indices], quota)]
            memory[c] = {"images": torch.stack([current_data[i][0].clone() for i in chosen]),
                         "coordinates": [list(current_data.samples[i][:2]) for i in chosen]}
        means = exemplar_means(model, memory, args)
        test_data = HoustonPatchClassification(label_file=args.test_label_file,
            class_ids=seen, max_samples_per_class=args.max_test_samples_per_class, **dataset_args)
        cm = evaluate_nme(model, test_data, means, sorted(memory), args)
        values, base_initial = metrics_row(cm, seen, current, sessions[0], base_initial)
        row = {"session": session, **values, "train_samples": len(train_data),
               "current_train_samples": len(current_data), "replay_train_samples": old_memory_count,
               "test_samples": len(test_data), "input_channels": hsi.shape[2],
               "memory_quota": quota, "memory_count": sum(len(e["images"]) for e in memory.values()),
               "memory_counts": {str(c): len(e["images"]) for c, e in sorted(memory.items())},
               "eval_classes": list(seen), "current_classes": list(current), "smoke": args.smoke}
        rows.append(row)
        torch.save({"model": model.state_dict(), "args": vars(args), "session": session,
                    "seen_classes": list(seen), "metrics": row, "memory": memory,
                    "mean_class_ids": sorted(memory), "exemplar_means": means}, output / f"session_{session}.pth")
        np.save(output / f"confusion_session_{session}.npy", cm)
        (output / "metrics.json").write_text(json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8")
        print(f"  OA={values['oa']:.6f}; AA={values['aa']:.6f}; memory={row['memory_count']}", flush=True)
    # Convenience export; JSON/checkpoint/CM remain the audited sources of truth.
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved iCaRL adaptation to {output}", flush=True)


if __name__ == "__main__":
    main(get_args_parser().parse_args())
