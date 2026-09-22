"""Matched ER/Full controls with identical persistent image memory.

Independent loop using the unchanged engine's loss, prototypes and calibration.
Shared random-priority memory does not use model features or test outcomes.
"""
import copy
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

import main_houston_cls as engine
from main_icarl_hsi import MemoryDataset, metrics_row, reduce_memory
from datasets.houston_cls import HoustonPatchClassification, load_mat_array, parse_class_sessions

VARIANT_FLAGS = {
    "persistent_er": dict(use_prototypes=0, adaptive_prototypes=0,
                          use_prototype_offsets=0, lambda_old_logit_distill=0.),
    "persistent_kd": dict(use_prototypes=0, adaptive_prototypes=0,
                          use_prototype_offsets=0, lambda_old_logit_distill=.05),
    "persistent_multiproto": dict(use_prototypes=1, adaptive_prototypes=1,
                                  use_prototype_offsets=0, lambda_old_logit_distill=0.),
    "persistent_multiproto_kd": dict(use_prototypes=1, adaptive_prototypes=1,
                                     use_prototype_offsets=0, lambda_old_logit_distill=.05),
    "persistent_multiproto_ipoc": dict(use_prototypes=1, adaptive_prototypes=1,
                                       use_prototype_offsets=1, lambda_old_logit_distill=0.),
    "persistent_fixed_full": dict(use_prototypes=1, adaptive_prototypes=0,
                                  use_prototype_offsets=1, lambda_old_logit_distill=.05),
    "persistent_full": dict(use_prototypes=1, adaptive_prototypes=1,
                            use_prototype_offsets=1, lambda_old_logit_distill=.05),
    "persistent_pool_lse": dict(use_prototypes=1, adaptive_prototypes=1,
                                use_prototype_offsets=1, lambda_old_logit_distill=.05),
    "persistent_pool_lme": dict(use_prototypes=1, adaptive_prototypes=1,
                                use_prototype_offsets=1, lambda_old_logit_distill=.05),
}


def get_args_parser():
    p = engine.get_args_parser()
    p.add_argument("--persistent_variant", choices=VARIANT_FLAGS, required=True)
    p.add_argument("--memory_budget", type=int, required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--allow-cpu", action="store_true")
    return p


def validate_args(args):
    sessions = parse_class_sessions(args.sessions)
    flat = [c for s in sessions for c in s]
    if len(set(flat)) != len(flat) or sorted(flat) != list(range(1, args.num_classes + 1)):
        raise ValueError("Sessions must partition all benchmark class IDs")
    if args.memory_budget < args.num_classes or min(args.epochs, args.batch_size, args.threads) < 1:
        raise ValueError("Invalid memory budget or training dimensions")
    if args.seed < 0 or args.hsi_patch_size != 15:
        raise ValueError("Nonnegative seed and predeclared patch15 required")
    if (args.model, args.mode, args.incremental_train, args.prompt_fusion) != (
            "pdp", "incremental", "replay", "joint"):
        raise ValueError("This control supports only the predeclared joint PDP learner")
    disabled = ("use_lidar", "use_spectral_prompt", "use_spectral_prototypes",
                "lambda_spectral_consistency", "feature_replay_filter", "spectral_replay_filter",
                "use_capst", "use_role_aware_offsets", "use_role_aware_adaptive_k",
                "use_confusion_aware_offsets", "use_prototype_offset_gates", "use_prototype_reliability",
                "use_role_aware_offset_l2", "use_balanced_offset_loss", "prompt_param_fusion", "prototype_ema")
    if any(getattr(args, key) for key in disabled):
        raise ValueError("Unspecified advanced modules are refused")
    if any(getattr(args, key) != value for key, value in VARIANT_FLAGS[args.persistent_variant].items()):
        raise ValueError("Component flags differ from the declared variant")
    pooling_variants = {"persistent_pool_lse": "logsumexp",
                        "persistent_pool_lme": "logmeanexp"}
    if (args.persistent_variant in pooling_variants and
            args.prototype_pooling != pooling_variants[args.persistent_variant]):
        raise ValueError("Pooling rule differs from the declared pooling variant")
    if args.smoke:
        if not (args.epochs == args.prototype_offset_epochs == 1 and
                0 < args.max_train_samples_per_class <= 8 and 0 < args.max_test_samples_per_class <= 12):
            raise ValueError("Smoke requires one epoch and capped samples")
    elif args.max_train_samples_per_class or args.max_test_samples_per_class:
        raise ValueError("Formal controls require complete splits")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA unavailable; no CPU fallback")
    if args.device == "cpu" and not args.smoke and not args.allow_cpu:
        raise RuntimeError("Formal CPU training refused")
    return sessions


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def image_memory_hash(memory):
    digest = hashlib.sha256()
    for c, entry in sorted(memory.items()):
        digest.update(str(c).encode())
        digest.update(str(tuple(entry["images"].shape)).encode())
        digest.update(entry["images"].contiguous().numpy().tobytes())
    return digest.hexdigest()


def coordinate_memory_hash(memory):
    return json_hash({str(c): entry["coordinates"] for c, entry in sorted(memory.items())})


def common_model_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if name.startswith(("features.", "prompt_pool.", "classifier.")):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def priority_order(samples, class_id, seed):
    indices = sorted((i for i, sample in enumerate(samples) if sample[2] == class_id),
                     key=lambda i: samples[i][:2])
    rng = np.random.default_rng(np.random.SeedSequence([seed, class_id, 20260918]))
    return [indices[int(i)] for i in rng.permutation(len(indices))]


def update_memory(memory, current_data, current_classes, quota, seed):
    memory = reduce_memory(memory, quota)
    for c in current_classes:
        if c in memory:
            raise ValueError("A seen class cannot be selected again")
        selected = priority_order(current_data.samples, c, seed)[:quota]
        if not selected:
            raise ValueError(f"Class {c} lacks training candidates")
        memory[c] = {"images": torch.stack([current_data[i][0].clone() for i in selected]),
                     "coordinates": [list(current_data.samples[i][:2]) for i in selected]}
    return memory


class PersistentTrainingDataset(Dataset):
    def __init__(self, current_data, memory, seen_classes):
        self.current = current_data
        self.replay = MemoryDataset(memory)
        self.samples = list(current_data.samples) + [
            (int(y), int(x), int(c)) for c, entry in sorted(memory.items())
            for y, x in entry["coordinates"]]
        self.class_ids = list(seen_classes)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.current[i] if i < len(self.current) else self.replay[i - len(self.current)]

    def class_counts(self):
        return {c: sum(sample[2] == c for sample in self.samples) for c in self.class_ids}


class EpochOrderSampler(Sampler):
    def __init__(self, length, seed, session):
        self.length, self.seed, self.session, self.order = length, seed, session, None

    def set_epoch(self, epoch):
        generator = torch.Generator().manual_seed(self.seed + 1000003 * self.session + 97 * epoch)
        self.order = torch.randperm(self.length, generator=generator).tolist()
        return json_hash(self.order)

    def __iter__(self):
        if self.order is None:
            raise RuntimeError("Set epoch before using training sampler")
        return iter(self.order)

    def __len__(self):
        return self.length


def make_loader(dataset, args, seed, sampler=None, shuffle=False):
    return DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, shuffle=shuffle,
        num_workers=0, drop_last=False, pin_memory=args.device == "cuda",
        generator=torch.Generator().manual_seed(seed))


def build_model(args, channels):
    # Exactly the constructor argument mapping used by engine.main; no global patch.
    names = inspect.signature(engine.HoustonPDPClassifier.__init__).parameters
    values = {k: v for k, v in vars(args).items() if k in names and k != "in_channels"}
    return engine.HoustonPDPClassifier(channels, **values).to(args.device)


def main(args):
    sessions = validate_args(args)
    torch.set_num_threads(args.threads)
    engine.seed_everything(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if list(output.glob("session_*.pth")) or (output / "metrics.json").exists():
        raise FileExistsError("Training artifacts exist; use a fresh output directory")
    train_map = load_mat_array(args.train_label_file, args.hsi_label_key, ndim=2)
    test_map = load_mat_array(args.test_label_file, args.hsi_label_key, ndim=2)
    if train_map.shape != test_map.shape or ((train_map > 0) & (test_map > 0)).any():
        raise ValueError("Train/test center maps overlap or differ in shape")
    hsi = engine.load_houston_hsi(args.hsi_file, args.hsi_key,
        band_indices=args.hsi_band_indices, normalize_per_band=bool(args.hsi_normalize_per_band))
    model = build_model(args, hsi.shape[2])
    initial_hash = common_model_hash(model)
    memory, seen, rows, base_initial = {}, [], [], None
    dataset_args = dict(hsi_file=args.hsi_file, hsi_array=hsi, hsi_key=args.hsi_key,
        label_key=args.hsi_label_key, patch_size=args.hsi_patch_size, seed=args.seed,
        band_indices=args.hsi_band_indices, normalize_per_band=bool(args.hsi_normalize_per_band))
    for session, current in enumerate(sessions, 1):
        old = list(seen)
        seen = sorted(seen + current)
        teacher = copy.deepcopy(model).eval() if old and args.lambda_old_logit_distill else None
        if teacher is not None:
            for param in teacher.parameters():
                param.requires_grad_(False)
        model.set_prompt_task_context(old, current, session_idx=session)
        current_data = HoustonPatchClassification(label_file=args.train_label_file,
            class_ids=current, max_samples_per_class=args.max_train_samples_per_class, **dataset_args)
        train_data = PersistentTrainingDataset(current_data, memory, seen)
        replay_hash = image_memory_hash(memory)
        train_hash = json_hash(train_data.samples)
        sampler = EpochOrderSampler(len(train_data), args.seed, session)
        train_loader = make_loader(train_data, args, args.seed + session, sampler=sampler)
        diagnostic_loader = make_loader(train_data, args, args.seed + 20000 + session)
        params = engine.configure_trainable_parameters(model, args, session)
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
        weights = engine.build_class_weights(train_data, args.num_classes, args.device,
            enabled=args.class_weight == "balanced", gamma=args.class_weight_gamma)
        order_hashes, rng_seeds = [], []
        print(f"Session {session}: current={len(current_data)}; replay={len(train_data.replay)}; seen={seen}", flush=True)
        for epoch in range(1, args.epochs + 1):
            order_hashes.append(sampler.set_epoch(epoch))
            rng_seed = args.seed + 1000003 * session + 97 * epoch
            rng_seeds.append(rng_seed)
            torch.manual_seed(rng_seed)
            torch.cuda.manual_seed_all(rng_seed)
            loss, acc = engine.train_one_epoch(model, train_loader, optimizer, args.device, epoch,
                weights, train_class_ids=seen, prompt_loss_weight=args.lambda_prompt_ortho,
                confusion_margin_weight=args.lambda_confusion_margin, confusion_margin=args.confusion_margin,
                confusion_topk=args.confusion_topk, confusion_margin_target_class_ids=current,
                confusion_margin_mode=args.confusion_margin_mode, teacher_model=teacher,
                distill_class_ids=old, old_logit_distill_weight=args.lambda_old_logit_distill,
                old_logit_distill_temperature=args.old_logit_distill_temperature,
                old_logit_distill_scope=args.old_logit_distill_scope,
                freeze_backbone_stats=bool(session > 1 and args.freeze_backbone_after_base))
            if not np.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                print(f"  epoch {epoch:03}: loss={loss:.6f}; accuracy={acc:.6f}", flush=True)
        engine.refresh_prototypes(model, diagnostic_loader, args.device, seen,
            kmeans_iters=args.prototype_kmeans_iters, adaptive_prototypes=bool(args.adaptive_prototypes),
            adaptive_min_prototypes=args.adaptive_min_prototypes, adaptive_k_mode=args.adaptive_k_mode,
            adaptive_elbow_min_gain=args.adaptive_elbow_min_gain)
        if args.use_prototype_offsets:
            model.reset_prototype_offsets(seen)
            calibration_loader = make_loader(train_data, args, args.seed + 30000 + session, shuffle=True)
            engine.calibrate_prototype_offsets(model, calibration_loader, args.device, seen, weights,
                epochs=args.prototype_offset_epochs, lr=args.prototype_offset_lr,
                l2_weight=args.prototype_offset_l2, prototype_alpha=args.prototype_alpha,
                prototype_temperature=args.prototype_temperature, prototype_pooling=args.prototype_pooling)
        test_data = HoustonPatchClassification(label_file=args.test_label_file, class_ids=seen,
            max_samples_per_class=args.max_test_samples_per_class, **dataset_args)
        test_loader = make_loader(test_data, args, args.seed + 40000 + session)
        cm = engine.evaluate(model, test_loader, args.device, allowed_class_ids=seen,
            num_classes=args.num_classes, use_prototypes=bool(args.use_prototypes),
            prototype_alpha=args.prototype_alpha, prototype_temperature=args.prototype_temperature,
            prototype_pooling=args.prototype_pooling)
        values, base_initial = metrics_row(cm, seen, current, sessions[0], base_initial)
        quota = args.memory_budget // len(seen)
        # Selection is training-only and model independent; test values do not enter it.
        memory = update_memory(memory, current_data, current, quota, args.seed)
        k = engine.active_prototype_k_stats(model, seen)
        row = {"session": session, **values, "train_samples": len(train_data),
            "test_samples": len(test_data), "input_channels": hsi.shape[2],
            "current_train_samples": len(current_data), "replay_train_samples": len(train_data.replay),
            "memory_quota": quota, "memory_count": sum(len(e["images"]) for e in memory.values()),
            "memory_counts": {str(c): len(e["images"]) for c, e in sorted(memory.items())},
            "train_samples_sha256": train_hash, "replay_images_sha256": replay_hash,
            "memory_images_sha256": image_memory_hash(memory),
            "memory_coordinates_sha256": coordinate_memory_hash(memory),
            "epoch_order_sha256s": order_hashes, "epoch_rng_seeds": rng_seeds,
            "initial_common_model_sha256": initial_hash,
            "active_prototype_k_by_class": k["by_class"],
            "active_prototype_k_mean": k["mean"], "smoke": args.smoke}
        rows.append(row)
        torch.save({"model": model.state_dict(), "args": vars(args), "session": session,
                    "seen_classes": seen, "metrics": row, "memory": memory}, output / f"session_{session}.pth")
        np.save(output / f"confusion_session_{session}.npy", cm)
        (output / "metrics.json").write_text(json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8")
        print(f"  OA={values['oa']:.6f}; AA={values['aa']:.6f}; memory={row['memory_count']}", flush=True)


if __name__ == "__main__":
    main(get_args_parser().parse_args())
