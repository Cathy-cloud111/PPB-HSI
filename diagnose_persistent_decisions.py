"""Frozen final-checkpoint decision audit. No training or parameter selection."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parent
MODES = ("head", "raw_fusion", "ipoc_fusion")
CONTRASTS = (("head", "raw_fusion"), ("raw_fusion", "ipoc_fusion"),
             ("head", "ipoc_fusion"))


def stage_groups(sessions, num_classes):
    flat = [c for stage in sessions for c in stage]
    if len(sessions) != 3 or sorted(flat) != list(range(1, num_classes + 1)):
        raise ValueError("Exactly three disjoint stages must partition all classes")
    return {"all_seen": sorted(flat), "base": sessions[0],
            "middle_increment": sessions[1], "last_increment": sessions[2]}


def confusion(targets, predictions, num_classes):
    import numpy as np
    targets, predictions = np.asarray(targets), np.asarray(predictions)
    if targets.shape != predictions.shape or targets.ndim != 1 or (
            not np.issubdtype(targets.dtype, np.integer) or
            not np.issubdtype(predictions.dtype, np.integer)):
        raise ValueError("Integer one-dimensional aligned labels required")
    if ((targets < 0) | (targets >= num_classes) |
            (predictions < 0) | (predictions >= num_classes)).any():
        raise ValueError("Prediction/target outside class range")
    return np.bincount(targets * num_classes + predictions,
                       minlength=num_classes ** 2).reshape(num_classes, num_classes)


def require_exact_confusion(actual, saved):
    import numpy as np
    if not np.array_equal(actual, saved):
        raise ValueError("Full confusion NOT reproduced exactly; stop, do not use this analysis")


def changes(targets, before, after, class_ids):
    import numpy as np
    mask = np.isin(targets, np.asarray(class_ids) - 1)
    y, a, b = targets[mask], before[mask], after[mask]
    n = len(y)
    correct_a, correct_b = a == y, b == y
    corrected = int((~correct_a & correct_b).sum())
    damaged = int((correct_a & ~correct_b).sum())
    both_wrong = ~correct_a & ~correct_b
    return {"support": n, "corrected": corrected, "damaged": damaged,
            "both_correct": int((correct_a & correct_b).sum()),
            "both_wrong_changed": int((both_wrong & (a != b)).sum()),
            "both_wrong_unchanged": int((both_wrong & (a == b)).sum()),
            "prediction_changed": int((a != b).sum()),
            "before_oa": float(correct_a.mean()) if n else None,
            "after_oa": float(correct_b.mean()) if n else None,
            "corrected_pp": 100 * corrected / n if n else None,
            "damaged_pp": 100 * damaged / n if n else None,
            "net_oa_pp": 100 * (corrected - damaged) / n if n else None}


def summarize_predictions(targets, predictions, groups, num_classes):
    import numpy as np
    from datasets.houston_cls import confusion_to_metrics
    matrices = {mode: confusion(targets, predictions[mode], num_classes) for mode in MODES}
    scores = {}
    for mode, cm in matrices.items():
        scores[mode] = {}
        for name, classes in groups.items():
            metric = confusion_to_metrics(cm, classes)
            scores[mode][name] = {k: metric[k] for k in ("oa", "aa", "total", "correct")}
    contrasts = {}
    for before, after in CONTRASTS:
        a, b = predictions[before], predictions[after]
        transitions = np.bincount(targets * num_classes ** 2 + a * num_classes + b,
                                 minlength=num_classes ** 3).reshape((num_classes,) * 3)
        contrasts[f"{before}_to_{after}"] = {
            "groups": {name: changes(targets, a, b, classes) for name, classes in groups.items()},
            "per_class": {str(c): changes(targets, a, b, [c]) for c in groups["all_seen"]},
            "changed_prediction_transitions": [
                {"true_class": int(y + 1), "before_class": int(p + 1),
                 "after_class": int(q + 1), "count": int(transitions[y, p, q])}
                for y, p, q in np.argwhere(transitions > 0) if p != q]}
    return {"scores": scores, "contrasts": contrasts,
            "confusions": {k: v.tolist() for k, v in matrices.items()}}


def state_hash(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str((tuple(value.shape), value.dtype)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def validate_source(plan, queue, results, pairs, smoke):
    import run_capacity_experiments as provenance
    seeds = (42,) if smoke else provenance.SEEDS
    expected = provenance.jobs_for(provenance.DATASETS, seeds, ("persistent_er", "persistent_full"))
    names = {j["name"] for j in expected}
    if any(x.get("smoke") is not smoke for x in (plan, queue, results, pairs)):
        raise ValueError("Smoke/formal identities differ")
    if queue.get("state") != "complete" or results.get("complete") is not True:
        raise ValueError("Original study is incomplete")
    for actual in (queue["completed"], [r["job"]["name"] for r in plan["requests"]],
                   [r["name"] for r in results["runs"]]):
        if len(actual) != len(names) or set(actual) != names:
            raise ValueError("Source must contain precisely the predeclared paired study")
    actual_pairs = [(p["dataset"], p["seed"]) for p in pairs["pairs"]]
    wanted_pairs = {(d, s) for d in provenance.DATASETS for s in seeds}
    if len(actual_pairs) != len(wanted_pairs) or set(actual_pairs) != wanted_pairs or any(
            p.get("passed") is not True for p in pairs["pairs"]):
        raise ValueError("Original reported pair checks failed/missing")
    return [r for r in plan["requests"] if r["job"]["variant"] == "persistent_full"]


def aggregate(runs):
    import numpy as np
    result = {}
    for dataset in sorted({r["job"]["dataset"] for r in runs}):
        selected = [r for r in runs if r["job"]["dataset"] == dataset]
        result[dataset] = {}
        for contrast in selected[0]["contrasts"]:
            result[dataset][contrast] = {}
            for group in selected[0]["groups"]:
                values = {}
                for key in ("corrected_pp", "damaged_pp", "net_oa_pp"):
                    raw = [r["contrasts"][contrast]["groups"][group][key] for r in selected]
                    values[key] = {"per_seed": raw, "mean": float(np.mean(raw)),
                                   "sample_sd": float(np.std(raw, ddof=1)) if len(raw) > 1 else None}
                result[dataset][contrast][group] = values
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true")
    for flag in ("input-root", "output-root", "houston-data-dir", "pavia-data-dir"):
        p.add_argument("--" + flag, type=Path)
    p.add_argument("--gpu")
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--threads", type=int, default=4)
    return p


def main():
    cli = parser().parse_args()
    if cli.threads < 1:
        raise ValueError("Positive thread count required")
    if cli.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cli.gpu
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = str(cli.threads)
    import numpy as np
    import scipy
    import torch
    from types import SimpleNamespace
    import main_persistent_controls as entry
    import persistent_control_audit as audit
    import run_capacity_experiments as provenance
    from datasets.houston_cls import HoustonPatchClassification, parse_class_sessions
    engine = entry.engine
    torch.set_num_threads(cli.threads)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if cli.device == "auto" else cli.device
    if device == "cuda" and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError("Exactly one visible GPU required; specify --gpu")
    runtime = {"python": platform.python_version(), "torch": str(torch.__version__),
               "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform(),
               "device": device, "threads": cli.threads,
               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    source = (cli.input_root or ROOT / ("persistent_smoke_outputs" if cli.smoke else "persistent_outputs")).resolve()
    output = (cli.output_root or ROOT / ("persistent_decision_smoke_outputs" if cli.smoke else "persistent_decision_outputs")).resolve()
    read = lambda name: json.loads((source / name).read_text(encoding="utf-8"))
    plan, queue, results, pairs = [read(n) for n in (
        "study_plan.json", "queue_status.json", "study_results.json", "pair_checks.json")]
    requests = validate_source(plan, queue, results, pairs, cli.smoke)
    if output.exists():
        raise FileExistsError("Output root exists; preserve it and choose a fresh --output-root")
    # Refuse different environments rather than silently relaxing reproduction checks.
    for request in plan["requests"]:
        identity = request["identity"]
        if identity["runtime"] != runtime:
            raise ValueError(f"Training/inference runtime differs: saved={identity['runtime']}; current={runtime}")
        if request["fingerprint"] != provenance.fingerprint(identity) or (
                identity["protocol"] != audit.scientific_args(request["effective_args"])):
            raise ValueError("Source identity is internally inconsistent")
        for relative, digest in identity["code_sha256"].items():
            if provenance.digest(ROOT / relative) != digest:
                raise ValueError(f"Training source changed: {relative}")
    output.mkdir(parents=True, exist_ok=False)
    state = {"state": "running", "smoke": cli.smoke, "started_at": provenance.now(),
             "total_jobs": len(requests), "completed": [], "runtime": runtime}
    provenance.save_json(output / "queue_status.json", state)
    source_hashes = {n: provenance.digest(source / n) for n in (
        "study_plan.json", "queue_status.json", "study_results.json", "pair_checks.json")}
    runs, scenes = [], {}
    try:
        for index, original in enumerate(requests, 1):
            request = copy.deepcopy(original)
            job, args_dict = request["job"], request["effective_args"]
            folder = source / job["name"]
            print(f"Auditing and inferring {index}/{len(requests)}: {job['name']}", flush=True)
            override = cli.houston_data_dir if job["dataset"] == "Houston2013" else cli.pavia_data_dir
            if override:
                flags = provenance.dataset_args(job["dataset"], cli.houston_data_dir, cli.pavia_data_dir)
                for i in (0, 2, 4):
                    args_dict[flags[i][2:]] = flags[i + 1]
            for flag, digest in request["identity"]["data_sha256"].items():
                if provenance.digest(Path(args_dict[flag[2:]])) != digest:
                    raise ValueError(f"Dataset content changed: {flag}")
            checked = audit.require_resume(folder, request)
            recorded = next(r for r in results["runs"] if r["name"] == job["name"])
            if checked != recorded["sessions"]:
                raise ValueError("Study summary differs from audited raw artifacts")
            er_request = copy.deepcopy(next(r for r in plan["requests"] if
                r["job"]["dataset"] == job["dataset"] and r["job"]["seed"] == job["seed"] and
                r["job"]["variant"] == "persistent_er"))
            for key in ("hsi_file", "train_label_file", "test_label_file"):
                er_request["effective_args"][key] = args_dict[key]
            er_checked = audit.require_resume(source / er_request["job"]["name"], er_request)
            er_recorded = next(r for r in results["runs"] if r["name"] == er_request["job"]["name"])
            if er_checked != er_recorded["sessions"]:
                raise ValueError("ER summary differs from audited raw artifacts")
            audit.check_pair(er_request, request, er_checked, checked)
            args = SimpleNamespace(**args_dict)
            entry.validate_args(args)
            sessions = parse_class_sessions(args.sessions)
            groups = stage_groups(sessions, args.num_classes)
            seen = groups["all_seen"]
            checkpoint = folder / "session_3.pth"
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if saved["session"] != 3 or saved["seen_classes"] != seen:
                raise ValueError("Final checkpoint class/session mismatch")
            engine.seed_everything(args.seed)
            scene_key = (args.hsi_file, args.hsi_key, args.hsi_band_indices, args.hsi_normalize_per_band)
            if scene_key not in scenes:
                scenes[scene_key] = engine.load_houston_hsi(args.hsi_file, args.hsi_key,
                    band_indices=args.hsi_band_indices, normalize_per_band=bool(args.hsi_normalize_per_band))
            scene = scenes[scene_key]
            model = entry.build_model(args, scene.shape[2])
            model.load_state_dict(saved["model"], strict=True)
            model.set_prompt_task_context(sorted(sessions[0] + sessions[1]), sessions[2], session_idx=3)
            model.eval()
            before_hash = state_hash(model)
            data = HoustonPatchClassification(hsi_file=args.hsi_file, hsi_array=scene,
                hsi_key=args.hsi_key, label_file=args.test_label_file, label_key=args.hsi_label_key,
                patch_size=args.hsi_patch_size, seed=args.seed, class_ids=seen,
                max_samples_per_class=args.max_test_samples_per_class,
                band_indices=args.hsi_band_indices, normalize_per_band=bool(args.hsi_normalize_per_band))
            loader = entry.make_loader(data, args, args.seed + 40000 + 3)
            targets, predicted = [], {m: [] for m in MODES}
            with torch.inference_mode():
                for batch_index, (images, labels) in enumerate(loader, 1):
                    images = images.to(device, non_blocking=True)
                    head, features, _ = engine.forward_for_classes(model, images, seen, return_features=True)
                    logits = {"head": head}
                    for mode, calibrated in (("raw_fusion", False), ("ipoc_fusion", True)):
                        logits[mode] = engine.add_prototype_logits(model, head, features, seen,
                            alpha=args.prototype_alpha, temperature=args.prototype_temperature,
                            pooling=args.prototype_pooling, calibrated=calibrated)
                    for mode, value in logits.items():
                        if not torch.isfinite(value).all():
                            raise ValueError("Nonfinite scores")
                        predicted[mode].append(engine.mask_logits_to_classes(value, seen).argmax(1).cpu().numpy())
                    targets.append(labels.numpy())
                    if batch_index % 25 == 0:
                        print(f"  batches {batch_index}/{len(loader)}", flush=True)
            targets = np.concatenate(targets)
            predicted = {m: np.concatenate(v) for m, v in predicted.items()}
            details = summarize_predictions(targets, predicted, groups, args.num_classes)
            require_exact_confusion(np.asarray(details["confusions"]["ipoc_fusion"]),
                                    np.load(folder / "confusion_session_3.npy", allow_pickle=False))
            if state_hash(model) != before_hash:
                raise ValueError("Inference changed model state")
            if provenance.digest(checkpoint) != checked[-1]["checkpoint_sha256"] or (
                    provenance.digest(folder / "confusion_session_3.npy") != checked[-1]["confusion_sha256"]):
                raise ValueError("Source artifacts changed during inference")
            run = {"job": job, "groups": groups, "full_confusion_reproduced_exactly": True,
                   "model_state_unchanged": True, "model_state_sha256": before_hash,
                   "test_coordinates_sha256": entry.json_hash(data.samples),
                   "checkpoint_sha256": checked[-1]["checkpoint_sha256"],
                   "source_fingerprint": original["fingerprint"],
                   "source_identity": original["identity"],
                   "original_pair_reaudited": True, **details}
            provenance.save_json(output / f"{job['name']}.json", run)
            runs.append(run)
            state["completed"].append(job["name"])
            provenance.save_json(output / "queue_status.json", state)
            delta = details["contrasts"]["raw_fusion_to_ipoc_fusion"]["groups"]["all_seen"]
            print(f"  Full reproduced exactly; IPOC corrected={delta['corrected']}, damaged={delta['damaged']}", flush=True)
            del model, saved, data, loader
        for name, digest in source_hashes.items():
            if provenance.digest(source / name) != digest:
                raise ValueError("Source study metadata changed during analysis")
        report = {"smoke": cli.smoke, "complete": True, "runtime": runtime,
                  "source_metadata_sha256": source_hashes,
                  "script_sha256": provenance.digest(Path(__file__)), "runs": runs,
                  "aggregate": aggregate(runs), "optimization_steps": 0,
                  "interpretation": "Post-hoc inference contrasts within frozen Full models. Not training-component causal effects, K/IPOC synergy, independent validation, or proof of novelty. No test-based parameter selection."}
        provenance.save_json(output / "decision_report.json", report)
        state.update(state="complete", finished_at=provenance.now())
        provenance.save_json(output / "queue_status.json", state)
        print(f"Complete: {output / 'decision_report.json'}", flush=True)
    except BaseException as error:
        state.update(state="failed", error=str(error), finished_at=provenance.now())
        provenance.save_json(output / "queue_status.json", state)
        raise


if __name__ == "__main__":
    main()
