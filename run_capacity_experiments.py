"""Predeclared capacity study; fresh runs, audited outputs, safe resume.

This does not change the training engine or select thresholds on test accuracy.
Smoke runs are explicitly separate and must never enter the paper's results.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
DATASETS = ("Houston2013", "PaviaU")
SEEDS = (42, 123, 3407)
VARIANTS = {
    **{f"fixed_k{k}": (False, k, .03, "logsumexp") for k in (2, 3, 5)},
    **{f"adaptive_eta{tag}_{pool}": (True, 5, eta, pooling)
       for tag, eta in (("003", .03), ("010", .10), ("020", .20))
       for pool, pooling in (("lse", "logsumexp"), ("lme", "logmeanexp"))},
}
METRICS = ("oa", "aa", "kappa", "current_oa", "base_oa",
           "apd_base_oa_raw", "apd_base_forgetting")


def now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def jobs_for(datasets, seeds, variants):
    return [{"name": f"{d}_{v}_seed{s}", "dataset": d, "variant": v, "seed": s}
            for d in datasets for s in seeds for v in variants]


def dataset_args(dataset, houston=None, pavia=None):
    if dataset == "Houston2013":
        data = Path(houston) if houston else ROOT / "data" / "Houston2013"
        files = ("HSI.mat", "TRLabel.mat", "TSLabel.mat")
        extra = ["--hsi_key", "HSI", "--sessions", "1-9,10-12,13-15", "--num_classes", "15"]
    else:
        data = Path(pavia) if pavia else ROOT / "data" / "PaviaU"
        files = ("PaviaU.mat", "PaviaU_train_spatial.mat", "PaviaU_test_spatial.mat")
        extra = ["--hsi_key", "paviaU", "--hsi_label_key", "paviaU_gt",
                 "--sessions", "1-5,6-7,8-9", "--num_classes", "9"]
    return [item for flag, file in zip(
        ("--hsi_file", "--train_label_file", "--test_label_file"), files)
        for item in (flag, str(data / file))] + extra


def build_command(job, output, device, houston=None, pavia=None, smoke=False):
    adaptive, k, eta, pooling = VARIANTS[job["variant"]]
    command = [sys.executable, "-u", str(ROOT / "main_houston_cls.py")]
    command += dataset_args(job["dataset"], houston, pavia)
    command += ["--model", "pdp", "--incremental_train", "replay",
                "--replay_old_samples_per_class", "100", "--hsi_patch_size", "15",
                "--epochs", "1" if smoke else "80", "--batch_size", "128",
                "--num_workers", "0", "--device", device,
                "--use_prototypes", "1", "--num_prototypes_per_class", str(k),
                "--adaptive_prototypes", str(int(adaptive)), "--adaptive_min_prototypes", "2",
                "--adaptive_k_mode", "elbow", "--adaptive_elbow_min_gain", str(eta),
                "--prototype_pooling", pooling, "--prototype_alpha", "0.7",
                "--prototype_temperature", "0.2", "--prototype_kmeans_iters", "10",
                "--use_prototype_offsets", "1", "--prototype_offset_scale", "0.02",
                "--prototype_offset_epochs", "1" if smoke else "10",
                "--prototype_offset_lr", "0.001", "--prototype_offset_l2", "0.01",
                "--prototype_offset_start_session", "2",
                "--lambda_old_logit_distill", "0.05",
                "--old_logit_distill_temperature", "2.0", "--old_logit_distill_scope", "old",
                "--freeze_backbone_after_base", "1", "--freeze_shared_prompts_after_base", "1",
                "--seed", str(job["seed"]), "--output_dir", str(output / job["name"])]
    if smoke:
        command += ["--max_train_samples_per_class", "8", "--max_test_samples_per_class", "12"]
    return command


def scientific_args(args):
    # Absolute paths and placement are not scientific hyperparameters. Data content
    # and code are separately hashed, so moving a result folder is not a new study.
    ignored = {"hsi_file", "train_label_file", "test_label_file", "output_dir", "device", "num_workers"}
    # The engine adds sample_filter=None at runtime even with all filtering
    # disabled. A NON-null filter remains part of the checked protocol.
    return {k: v for k, v in args.items() if k not in ignored
            and not (k == "sample_filter" and v is None)}


def check_close(actual, expected, description):
    if not math.isfinite(float(actual)) or not math.isclose(
            float(actual), float(expected), rel_tol=1e-8, abs_tol=1e-10):
        raise ValueError(f"{description}: recorded={actual}, recomputed={expected}")


def audit_run(folder, expected):
    import numpy as np
    import torch
    from datasets.houston_cls import confusion_to_metrics, parse_class_sessions

    rows = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    if [r["session"] for r in rows] != [1, 2, 3]:
        raise ValueError(f"{folder}: expected exactly three sessions")
    sessions = parse_class_sessions(expected["sessions"])
    seen, base_initial, result = [], None, []
    for idx, (row, current) in enumerate(zip(rows, sessions), 1):
        saved = torch.load(folder / f"session_{idx}.pth", map_location="cpu", weights_only=True)
        if scientific_args(saved["args"]) != scientific_args(expected):
            raise ValueError(f"{folder}: checkpoint {idx} has different training arguments")
        seen += list(current)
        if saved["session"] != idx or saved["seen_classes"] != seen:
            raise ValueError(f"{folder}: checkpoint/session class mismatch")
        cm = np.load(folder / f"confusion_session_{idx}.npy", allow_pickle=False)
        if cm.shape != (expected["num_classes"], expected["num_classes"]) or (
                not np.isfinite(cm).all() or (cm < 0).any() or not np.equal(cm, np.floor(cm)).all()):
            raise ValueError(f"{folder}: invalid confusion matrix")
        metrics = confusion_to_metrics(cm, seen)
        base = confusion_to_metrics(cm, sessions[0])["oa"]
        if idx == 1:
            base_initial = base
        values = {k: metrics[k] for k in ("oa", "aa", "kappa")}
        values.update(current_oa=confusion_to_metrics(cm, current)["oa"], base_oa=base,
                      apd_base_oa_raw=base_initial - base,
                      apd_base_forgetting=max(0., base_initial - base))
        for key, value in values.items():
            check_close(row[key], value, f"{folder.name}/session{idx}/{key}")
            check_close(saved["metrics"][key], value, f"checkpoint/{key}")
        if int(row["test_samples"]) != int(cm.sum()):
            raise ValueError(f"{folder}: test sample count does not match confusion")
        counts = saved["model"]["prototype_counts"]
        if counts.shape != (expected["num_classes"], expected["num_prototypes_per_class"]) or (
                not torch.isfinite(counts).all() or (counts < 0).any()):
            raise ValueError(f"{folder}: invalid prototype count tensor")
        by_class = {str(c): int((counts[c - 1] > 0).sum()) for c in seen}
        kmax = expected["num_prototypes_per_class"]
        # K is requested cluster capacity; an empty fitted cluster is inactive.
        # Report actual active K rather than falsely assuming every slot is used.
        if not all(1 <= k <= kmax for k in by_class.values()):
            raise ValueError(f"{folder}: prototype capacity outside configured range")
        unseen = sorted(set(range(1, expected["num_classes"] + 1)) - set(seen))
        if unseen and (counts[[c - 1 for c in unseen]] > 0).any():
            raise ValueError(f"{folder}: unseen classes have active prototypes")
        if unseen and (cm[[c - 1 for c in unseen]].sum() or cm[:, [c - 1 for c in unseen]].sum()):
            raise ValueError(f"{folder}: confusion contains unseen classes")
        if by_class != json.loads(row["active_prototype_k_by_class"]):
            raise ValueError(f"{folder}: reported K differs from checkpoint")
        mean = sum(by_class.values()) / len(by_class)
        check_close(row["active_prototype_k_mean"], mean, "active K mean")
        result.append({"session": idx, **values, "active_k_by_class": by_class,
                       "active_k_mean": mean, "active_prototypes": sum(by_class.values()),
                       "allocated_prototype_slots": counts.numel(),
                       "checkpoint_sha256": digest(folder / f"session_{idx}.pth"),
                       "confusion_sha256": digest(folder / f"confusion_session_{idx}.npy")})
    return result


def require_resume(folder, request):
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest["fingerprint"] != request["fingerprint"]:
        raise ValueError(f"{folder}: protocol/code/data/runtime changed; use a new --output-root")
    if not (folder / "completed.json").exists():
        raise ValueError(f"{folder}: unfinished run; preserve it and use a new --output-root")
    done = json.loads((folder / "completed.json").read_text(encoding="utf-8"))
    if done["fingerprint"] != request["fingerprint"]:
        raise ValueError(f"{folder}: completion fingerprint mismatch")
    audited = audit_run(folder, request["effective_args"])
    if done["sessions"] != audited:
        raise ValueError(f"{folder}: completed outputs changed after original audit")
    return audited


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    p.add_argument("--houston-data-dir", type=Path)
    p.add_argument("--pavia-data-dir", type=Path)
    p.add_argument("--output-root", type=Path)
    p.add_argument("--gpu", help="Physical GPU ID(s), set before importing torch; normally one ID")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--allow-cpu", action="store_true", help="Explicitly authorize long formal CPU training")
    p.add_argument("--threads", type=int, default=4)
    return p


def main():
    args = parser().parse_args()
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    import numpy as np
    import scipy
    import torch
    import main_houston_cls as engine

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if not args.dry_run:
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if device == "cuda" and torch.cuda.device_count() > 1:
            raise RuntimeError("Select one GPU explicitly with --gpu; this is a sequential single-GPU queue")
        if device == "cpu" and not args.smoke and not args.allow_cpu:
            raise RuntimeError("Formal CPU training requires --allow-cpu; use the server or --smoke instead")
    output = (args.output_root or ROOT / ("capacity_smoke_outputs" if args.smoke else "capacity_outputs")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    jobs = jobs_for(args.datasets, args.seeds, args.variants)
    if len({j["name"] for j in jobs}) != len(jobs):
        raise ValueError("Repeated datasets/seeds/variants would duplicate jobs")
    hashes = {}
    for dataset in args.datasets:
        file_args = dataset_args(dataset, args.houston_data_dir, args.pavia_data_dir)
        hashes[dataset] = {file_args[i]: digest(Path(file_args[i + 1])) for i in (0, 2, 4)}
    code = {str(p.relative_to(ROOT)): digest(p) for p in (
        ROOT / "main_houston_cls.py", ROOT / "datasets" / "houston_cls.py",
        ROOT / "datasets" / "__init__.py")}
    runtime = {"python": platform.python_version(), "torch": str(torch.__version__),
               "numpy": np.__version__, "scipy": scipy.__version__,
               "platform": platform.platform(), "device": device, "threads": args.threads,
               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    requests = []
    for job in jobs:
        command = build_command(job, output, device, args.houston_data_dir, args.pavia_data_dir, args.smoke)
        effective = vars(engine.get_args_parser().parse_args(command[3:]))
        identity = {"smoke": args.smoke, "protocol": scientific_args(effective),
                    "code_sha256": code, "data_sha256": hashes[job["dataset"]], "runtime": runtime}
        requests.append({"job": job, "fingerprint": fingerprint(identity),
                         "effective_args": effective, "identity": identity, "command": command})
    plan = {"created_at": now(), "smoke": args.smoke, "total_jobs": len(jobs),
            "runner_sha256": digest(Path(__file__)), "requests": requests,
            "interpretation": "Report the entire grid; do not select eta using test OA. Active K is not allocated memory."}
    if args.dry_run:
        save_json(output / "plan.json", plan)
        print(f"Plan only: {len(jobs)} jobs; no training started. {output / 'plan.json'}", flush=True)
        return
    lock = output / ".queue.lock"
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as exc:
        raise RuntimeError(f"Existing queue lock {lock}; inspect its PID before removing a stale lock") from exc
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "started_at": now()}, handle)
    state = {"started_at": now(), "runner_pid": os.getpid(), "smoke": args.smoke,
             "state": "running", "total_jobs": len(jobs), "completed": [], "runtime": runtime}
    results, process = [], None
    try:
        # Reject incompatible/unfinished outputs before launching any new training.
        existing = {}
        for request in requests:
            folder = output / request["job"]["name"]
            if folder.exists() and any(folder.iterdir()):
                existing[folder.name] = require_resume(folder, request)
        save_json(output / "study_plan.json", plan)
        save_json(output / "queue_status.json", state)
        for request in requests:
            job = request["job"]
            folder = output / job["name"]
            if job["name"] in existing:
                sessions = existing[job["name"]]
                elapsed = json.loads((folder / "completed.json").read_text())["elapsed_seconds"]
                print(f"Verified resume: {job['name']}", flush=True)
            else:
                folder.mkdir(exist_ok=True)
                save_json(folder / "manifest.json", request)
                save_json(folder / "command.json", request["command"])
                state.update(current_job=job, job_started_at=now())
                save_json(output / "queue_status.json", state)
                print(f"Starting {job['name']} ({len(state['completed'])}/{len(jobs)} done)", flush=True)
                start = time.monotonic()
                with (folder / "train.log").open("w", encoding="utf-8") as log:
                    process = subprocess.Popen(request["command"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    state["training_pid"] = process.pid
                    save_json(output / "queue_status.json", state)
                    code = process.wait()
                    process = None
                if code:
                    raise RuntimeError(f"{job['name']} exited {code}; see {folder / 'train.log'}")
                elapsed = time.monotonic() - start
                sessions = audit_run(folder, request["effective_args"])
                save_json(folder / "completed.json", {"finished_at": now(), "fingerprint": request["fingerprint"],
                          "elapsed_seconds": elapsed, "sessions": sessions})
            state["completed"].append(job["name"])
            state.pop("training_pid", None)
            results.append({**job, "elapsed_seconds": elapsed, "sessions": sessions})
            save_json(output / "study_results.json", {"smoke": args.smoke, "planned_jobs": len(jobs),
                      "complete": len(results) == len(jobs), "runs": results})
            save_json(output / "queue_status.json", state)
        state.update(state="complete", finished_at=now())
        state.pop("current_job", None)
        save_json(output / "queue_status.json", state)
        print(f"Complete: {len(jobs)} audited {'SMOKE (not paper evidence)' if args.smoke else 'formal'} runs", flush=True)
    except BaseException as exc:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=30)
        state.update(state="failed", error=str(exc), finished_at=now())
        save_json(output / "queue_status.json", state)
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
