"""Artifact and pair checks for shared persistent-memory ER/Full controls."""
import json


PAIR_FIELDS = ("train_samples", "test_samples", "current_train_samples", "replay_train_samples",
    "memory_quota", "memory_count", "memory_counts", "train_samples_sha256", "replay_images_sha256",
    "memory_images_sha256", "memory_coordinates_sha256", "epoch_order_sha256s", "epoch_rng_seeds",
    "initial_common_model_sha256")

COMPONENT_FIELDS = {"persistent_variant", "use_prototypes", "adaptive_prototypes",
                    "use_prototype_offsets", "lambda_old_logit_distill",
                    "prototype_pooling"}


def scientific_args(args):
    import run_capacity_experiments as provenance
    result = provenance.scientific_args(args)
    result.pop("allow_cpu", None)
    return result


def selected_current_samples(label_map, classes, cap, seed):
    from types import SimpleNamespace
    from datasets.houston_cls import HoustonPatchClassification
    # Dataset selection does not require image access, padding, or a scene reload.
    view = SimpleNamespace(labels=label_map, class_ids=sorted(classes))
    return HoustonPatchClassification._build_samples(view, max_samples_per_class=cap, seed=seed)


def audit_run(folder, expected):
    import numpy as np
    import torch
    from datasets.houston_cls import load_mat_array, parse_class_sessions
    import run_capacity_experiments as provenance
    import main_persistent_controls as entry
    checked = provenance.audit_run(folder, expected)
    rows = json.loads((folder / "metrics.json").read_text())
    train = load_mat_array(expected["train_label_file"], expected["hsi_label_key"], ndim=2)
    test = load_mat_array(expected["test_label_file"], expected["hsi_label_key"], ndim=2)
    sessions = parse_class_sessions(expected["sessions"])
    seen, previous = [], {}
    for idx, (row, current, result) in enumerate(zip(rows, sessions, checked), 1):
        saved = torch.load(folder / f"session_{idx}.pth", map_location="cpu", weights_only=True)
        if row != saved["metrics"] or row["smoke"] != expected["smoke"]:
            raise ValueError("Checkpoint metrics or smoke flags changed")
        old = list(seen)
        seen = sorted(seen + current)
        cm = np.load(folder / f"confusion_session_{idx}.npy", allow_pickle=False)
        for c in seen:
            total = int((test == c).sum())
            cap = expected["max_test_samples_per_class"]
            if int(cm[c - 1].sum()) != (min(cap, total) if cap else total):
                raise ValueError("Evaluation counts differ from supplied split")
        samples = selected_current_samples(train, current, expected["max_train_samples_per_class"], expected["seed"])
        all_samples = list(samples) + [(y, x, c) for c, data in sorted(previous.items())
                                      for y, x in data["coordinates"]]
        if row["train_samples_sha256"] != entry.json_hash(all_samples) or (
                row["current_train_samples"] != len(samples) or
                row["replay_train_samples"] != len(all_samples) - len(samples) or
                row["train_samples"] != len(all_samples)):
            raise ValueError("Actual current/replay sample list differs from declaration")
        if row["replay_images_sha256"] != entry.image_memory_hash(previous):
            raise ValueError("Replay image memory changed")
        quota = expected["memory_budget"] // len(seen)
        memory = saved["memory"]
        if set(memory) != set(seen) or row["memory_quota"] != quota:
            raise ValueError("Invalid memory quota/class set")
        for c, data in memory.items():
            images, coords = data["images"], data["coordinates"]
            if (images.ndim != 4 or images.shape[1:] != (row["input_channels"], 15, 15) or
                    images.dtype != torch.float32 or not torch.isfinite(images).all() or
                    len(images) != len(coords) or not 1 <= len(images) <= quota or
                    len({tuple(p) for p in coords}) != len(coords)):
                raise ValueError("Invalid persisted image memory")
            if any(not (0 <= y < train.shape[0] and 0 <= x < train.shape[1]) or train[y, x] != c for y, x in coords):
                raise ValueError("Memory includes an illegal training center")
            if c in old:
                if coords != previous[c]["coordinates"][:quota] or not torch.equal(images, previous[c]["images"][:quota]):
                    raise ValueError("Old image/coordinate prefix was altered")
            else:
                selected = entry.priority_order(samples, c, expected["seed"])[:quota]
                if coords != [list(samples[i][:2]) for i in selected]:
                    raise ValueError("New memory did not follow fixed training-only priorities")
        counts = {str(c): len(data["images"]) for c, data in sorted(memory.items())}
        if row["memory_counts"] != counts or row["memory_count"] != sum(counts.values()) or (
                row["memory_count"] > expected["memory_budget"]):
            raise ValueError("Memory count exceeds budget or disagrees with images")
        if row["memory_images_sha256"] != entry.image_memory_hash(memory) or (
                row["memory_coordinates_sha256"] != entry.coordinate_memory_hash(memory)):
            raise ValueError("Memory hash changed")
        sampler = entry.EpochOrderSampler(len(all_samples), expected["seed"], idx)
        orders = [sampler.set_epoch(epoch) for epoch in range(1, expected["epochs"] + 1)]
        rng_seeds = [expected["seed"] + 1000003 * idx + 97 * epoch for epoch in range(1, expected["epochs"] + 1)]
        if row["epoch_order_sha256s"] != orders or row["epoch_rng_seeds"] != rng_seeds:
            raise ValueError("Unexpected batch order or training RNG schedule")
        if len(row["initial_common_model_sha256"]) != 64:
            raise ValueError("Missing initial common model hash")
        result.update({key: row[key] for key in PAIR_FIELDS})
        previous = memory
    return checked


def require_resume(folder, request):
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["fingerprint"] != request["fingerprint"]:
        raise ValueError("Code/data/runtime/protocol changed; preserve outputs and use a fresh root")
    done_path = folder / "completed.json"
    if not done_path.exists():
        raise ValueError("Unfinished outputs preserved; use a fresh root")
    done = json.loads(done_path.read_text())
    if done["fingerprint"] != request["fingerprint"]:
        raise ValueError("Completion identity differs")
    result = audit_run(folder, request["effective_args"])
    if result != done["sessions"]:
        raise ValueError("Completed artifacts changed after audit")
    return result


def check_pair(er_request, full_request, er_sessions, full_sessions):
    import run_capacity_experiments as provenance
    for request in (er_request, full_request):
        if request["fingerprint"] != provenance.fingerprint(request["identity"]) or (
                request["identity"]["protocol"] != scientific_args(request["effective_args"])):
            raise ValueError("Invalid pair identity")
    if er_request["job"]["dataset"] != full_request["job"]["dataset"] or er_request["job"]["seed"] != full_request["job"]["seed"]:
        raise ValueError("Pair dataset/seed differs")
    for field in ("code_sha256", "data_sha256", "runtime", "smoke"):
        if er_request["identity"][field] != full_request["identity"][field]:
            raise ValueError(f"Pair {field} differs")
    a, b = [scientific_args(r["effective_args"]) for r in (er_request, full_request)]
    differences = {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
    if differences != {"persistent_variant", "use_prototypes", "adaptive_prototypes",
                       "use_prototype_offsets", "lambda_old_logit_distill"}:
        raise ValueError(f"Undeclared pair differences: {differences}")
    if er_request["effective_args"]["persistent_variant"] != "persistent_er" or (
            full_request["effective_args"]["persistent_variant"] != "persistent_full"):
        raise ValueError("Wrong pair configuration")
    if len(er_sessions) != 3 or len(full_sessions) != 3:
        raise ValueError("Incomplete pair")
    for a, b in zip(er_sessions, full_sessions):
        if a["session"] != b["session"] or any(a[key] != b[key] for key in PAIR_FIELDS):
            raise ValueError("Pair memory, input sequence, initialization, batch order or RNG differs")
    return {"dataset": er_request["job"]["dataset"], "seed": er_request["job"]["seed"],
            "passed": True, "checked_fields": list(PAIR_FIELDS)}


def check_matched_group(requests, sessions):
    """Verify that an ablation group differs only in declared component flags."""
    import run_capacity_experiments as provenance
    if len(requests) != len(sessions) or len(requests) < 2:
        raise ValueError("Matched group requires parallel requests and sessions")
    dataset = requests[0]["job"]["dataset"]
    seed = requests[0]["job"]["seed"]
    reference_identity = requests[0]["identity"]
    reference_args = scientific_args(requests[0]["effective_args"])
    for request, rows in zip(requests, sessions):
        if request["fingerprint"] != provenance.fingerprint(request["identity"]) or (
                request["identity"]["protocol"] != scientific_args(request["effective_args"])):
            raise ValueError("Invalid matched-group identity")
        if (request["job"]["dataset"], request["job"]["seed"]) != (dataset, seed):
            raise ValueError("Matched-group dataset/seed differs")
        for field in ("code_sha256", "data_sha256", "runtime", "smoke"):
            if request["identity"][field] != reference_identity[field]:
                raise ValueError(f"Matched-group {field} differs")
        actual = scientific_args(request["effective_args"])
        differences = {key for key in reference_args.keys() | actual.keys()
                       if reference_args.get(key) != actual.get(key)}
        if not differences <= COMPONENT_FIELDS:
            raise ValueError(f"Undeclared matched-group differences: {differences}")
        if len(rows) != 3:
            raise ValueError("Incomplete matched-group run")
    reference_rows = sessions[0]
    for rows in sessions[1:]:
        for left, right in zip(reference_rows, rows):
            if left["session"] != right["session"] or any(
                    left[field] != right[field] for field in PAIR_FIELDS):
                raise ValueError("Ablation memory, inputs, initialization, order or RNG differs")
    return {"dataset": dataset, "seed": seed, "passed": True,
            "variants": [request["job"]["variant"] for request in requests],
            "checked_fields": list(PAIR_FIELDS)}
