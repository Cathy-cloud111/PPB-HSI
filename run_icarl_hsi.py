"""Immutable six-run iCaRL HSI reference queue, not a budget-matched claim."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import run_capacity_experiments as provenance

ROOT = Path(__file__).resolve().parent


def build_command(job, output, device, houston=None, pavia=None, smoke=False, threads=4):
    classes = 15 if job["dataset"] == "Houston2013" else 9
    return [sys.executable, "-u", str(ROOT / "main_icarl_hsi.py")] + \
        provenance.dataset_args(job["dataset"], houston, pavia) + [
            "--memory_budget", str(classes * (4 if smoke else 100)),
            "--epochs", "1" if smoke else "80", "--batch_size", "128",
            "--lr", ".001", "--weight_decay", ".0001", "--dropout", ".2",
            "--hsi_patch_size", "15", "--hsi_normalize_per_band", "1",
            "--seed", str(job["seed"]), "--device", device, "--threads", str(threads),
            "--output_dir", str(output / job["name"])] + ([
            "--smoke", "--max_train_samples_per_class", "8",
            "--max_test_samples_per_class", "12"] if smoke else [])


def scientific_args(args):
    return {k: v for k, v in args.items() if k not in (
        "hsi_file", "train_label_file", "test_label_file", "output_dir", "device", "allow_cpu")}


def audit_run(folder, expected):
    import numpy as np
    import torch
    from datasets.houston_cls import confusion_to_metrics, load_mat_array, parse_class_sessions
    rows = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    sessions = parse_class_sessions(expected["sessions"])
    if [r["session"] for r in rows] != list(range(1, len(sessions) + 1)):
        raise ValueError("Incomplete or unordered sessions")
    train_map = load_mat_array(expected["train_label_file"], expected["hsi_label_key"], ndim=2)
    test_map = load_mat_array(expected["test_label_file"], expected["hsi_label_key"], ndim=2)
    seen, base_initial, previous_memory, result = [], None, {}, []
    for idx, (row, current) in enumerate(zip(rows, sessions), 1):
        saved = torch.load(folder / f"session_{idx}.pth", map_location="cpu", weights_only=True)
        if scientific_args(saved["args"]) != scientific_args(expected):
            raise ValueError("Checkpoint training arguments differ from manifest")
        old = list(seen)
        seen += current
        if saved["session"] != idx or saved["seen_classes"] != seen or row["eval_classes"] != seen:
            raise ValueError("Class/session mismatch")
        if row["current_classes"] != current or row["smoke"] != expected["smoke"]:
            raise ValueError("Current classes or smoke status differ")
        cm = np.load(folder / f"confusion_session_{idx}.npy", allow_pickle=False)
        n = expected["num_classes"]
        if cm.shape != (n, n) or not np.isfinite(cm).all() or (cm < 0).any() or not (cm == np.floor(cm)).all():
            raise ValueError("Invalid confusion matrix")
        unseen = sorted(set(range(1, n + 1)) - set(seen))
        if unseen and (cm[[c - 1 for c in unseen]].sum() or cm[:, [c - 1 for c in unseen]].sum()):
            raise ValueError("Unseen ground truth/predictions included")
        for c in seen:
            count = int((test_map == c).sum())
            cap = expected["max_test_samples_per_class"]
            if int(cm[c - 1].sum()) != (min(count, cap) if cap else count):
                raise ValueError("Test class count differs from supplied split")
        metrics = confusion_to_metrics(cm, seen)
        base = confusion_to_metrics(cm, sessions[0])["oa"]
        if base_initial is None:
            base_initial = base
        values = {k: metrics[k] for k in ("oa", "aa", "kappa")}
        values.update(current_oa=confusion_to_metrics(cm, current)["oa"], base_oa=base,
                      apd_base_oa_raw=base_initial - base,
                      apd_base_forgetting=max(0., base_initial - base))
        for k, value in values.items():
            provenance.check_close(row[k], value, f"{folder.name}/{idx}/{k}")
        if row != saved["metrics"] or row["test_samples"] != int(cm.sum()):
            raise ValueError("Metrics/checkpoint/test counts disagree")
        memory = saved["memory"]
        quota = expected["memory_budget"] // len(seen)
        if set(memory) != set(seen) or row["memory_quota"] != quota:
            raise ValueError("Memory class set or quota mismatch")
        for c, entry in memory.items():
            images, coords = entry["images"], entry["coordinates"]
            if (images.ndim != 4 or images.shape[1:] != (
                    row["input_channels"], expected["hsi_patch_size"], expected["hsi_patch_size"]) or
                    images.dtype != torch.float32 or not torch.isfinite(images).all() or
                    len(images) != len(coords) or not 1 <= len(images) <= quota or
                    len({tuple(p) for p in coords}) != len(coords)):
                raise ValueError("Invalid persisted exemplar images/coordinates")
            if any(not (0 <= y < train_map.shape[0] and 0 <= x < train_map.shape[1]) or
                   int(train_map[y, x]) != c for y, x in coords):
                raise ValueError("Exemplar not drawn from class training centers")
            if c in old:
                prior = previous_memory[c]
                if coords != prior["coordinates"][:quota] or not torch.equal(images, prior["images"][:quota]):
                    raise ValueError("Old memory reselected or changed instead of prefix reduction")
            else:
                candidates = int((train_map == c).sum())
                cap = expected["max_train_samples_per_class"]
                if len(images) != min(quota, min(candidates, cap) if cap else candidates):
                    raise ValueError("Unexpected new exemplar count")
        counts = {str(c): len(entry["images"]) for c, entry in sorted(memory.items())}
        memory_total = sum(counts.values())
        if row["memory_counts"] != counts or row["memory_count"] != memory_total or memory_total > expected["memory_budget"]:
            raise ValueError("Persistent memory budget/count mismatch")
        current_total = sum(min(int((train_map == c).sum()), expected["max_train_samples_per_class"])
                            if expected["max_train_samples_per_class"] else int((train_map == c).sum())
                            for c in current)
        replay_total = sum(len(entry["images"]) for entry in previous_memory.values())
        if (row["current_train_samples"], row["replay_train_samples"], row["train_samples"]) != (
                current_total, replay_total, current_total + replay_total):
            raise ValueError("Training did not use declared current data plus previous memory")
        means = saved["exemplar_means"]
        if saved["mean_class_ids"] != sorted(seen) or means.shape != (len(seen), 256) or (
                not torch.isfinite(means).all() or not torch.allclose(means.norm(dim=1), torch.ones(len(seen)), atol=1e-5)):
            raise ValueError("Invalid normalized NME means")
        result.append({"session": idx, **values, "memory_counts": counts,
                       "memory_count": memory_total, "train_samples": row["train_samples"],
                       "test_samples": row["test_samples"],
                       "checkpoint_sha256": provenance.digest(folder / f"session_{idx}.pth"),
                       "confusion_sha256": provenance.digest(folder / f"confusion_session_{idx}.npy")})
        previous_memory = memory
    return result


def require_resume(folder, request):
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["fingerprint"] != request["fingerprint"]:
        raise ValueError("Code/data/runtime/protocol changed; use a fresh --output-root")
    completion = folder / "completed.json"
    if not completion.exists():
        raise ValueError("Unfinished artifacts preserved; use a fresh --output-root")
    done = json.loads(completion.read_text())
    if done["fingerprint"] != request["fingerprint"]:
        raise ValueError("Completion fingerprint mismatch")
    sessions = audit_run(folder, request["effective_args"])
    if done["sessions"] != sessions:
        raise ValueError("Completed artifacts changed")
    return sessions


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", choices=provenance.DATASETS, default=list(provenance.DATASETS))
    p.add_argument("--seeds", nargs="+", type=int, default=list(provenance.SEEDS))
    for flag in ("houston-data-dir", "pavia-data-dir", "output-root"):
        p.add_argument("--" + flag, type=Path)
    p.add_argument("--gpu")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--allow-cpu", action="store_true")
    return p


def main():
    args = parser().parse_args()
    if args.threads < 1:
        raise ValueError("Positive thread count required")
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = str(args.threads)
    import numpy as np
    import scipy
    import torch
    import main_icarl_hsi as entry
    torch.set_num_threads(args.threads)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if not args.dry_run and device == "cuda" and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError("Exactly one visible GPU required; use --gpu")
    if not args.dry_run and device == "cpu" and not args.smoke and not args.allow_cpu:
        raise RuntimeError("Formal CPU training refused")
    output = (args.output_root or ROOT / ("icarl_smoke_outputs" if args.smoke else "icarl_outputs")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    jobs = provenance.jobs_for(args.datasets, args.seeds, ("icarl_hsi",))
    if len(jobs) != len({j["name"] for j in jobs}):
        raise ValueError("Duplicate jobs")
    code = {str(p.relative_to(ROOT)): provenance.digest(p) for p in (
        ROOT / "main_icarl_hsi.py", Path(__file__), ROOT / "main_houston_cls.py",
        ROOT / "run_capacity_experiments.py", ROOT / "datasets/houston_cls.py", ROOT / "datasets/__init__.py")}
    runtime = {"python": platform.python_version(), "torch": str(torch.__version__),
               "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform(),
               "device": device, "threads": args.threads,
               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    data = {}
    for dataset in args.datasets:
        flags = provenance.dataset_args(dataset, args.houston_data_dir, args.pavia_data_dir)
        data[dataset] = {flags[i]: provenance.digest(Path(flags[i + 1])) for i in (0, 2, 4)}
    requests = []
    for job in jobs:
        command = build_command(job, output, device, args.houston_data_dir, args.pavia_data_dir, args.smoke, args.threads)
        if args.allow_cpu:
            command.append("--allow-cpu")
        effective = vars(entry.get_args_parser().parse_args(command[3:]))
        identity = {"smoke": args.smoke, "protocol": scientific_args(effective),
                    "code_sha256": code, "data_sha256": data[job["dataset"]], "runtime": runtime}
        requests.append({"job": job, "command": command, "effective_args": effective,
                         "identity": identity, "fingerprint": provenance.fingerprint(identity)})
    plan = {"requests": requests, "total_jobs": len(jobs), "smoke": args.smoke,
            "interpretation": "HSI adaptation; persistent total memory vs original learner's historical resampling; not budget-matched."}
    if args.dry_run:
        provenance.save_json(output / "plan.json", plan)
        print(f"Plan only: {len(jobs)} runs; training not started.", flush=True)
        return
    lock = output / ".queue.lock"
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(fd, "w") as handle:
        json.dump({"pid": os.getpid(), "started_at": provenance.now()}, handle)
    state = {"started_at": provenance.now(), "runner_pid": os.getpid(), "smoke": args.smoke,
             "total_jobs": len(jobs), "completed": [], "state": "running", "runtime": runtime}
    results, process = [], None
    try:
        existing = {}
        for request in requests:
            folder = output / request["job"]["name"]
            if folder.exists() and any(folder.iterdir()):
                existing[folder.name] = require_resume(folder, request)
        provenance.save_json(output / "study_plan.json", plan)
        for request in requests:
            job, folder = request["job"], output / request["job"]["name"]
            if folder.name in existing:
                sessions = existing[folder.name]
                elapsed = json.loads((folder / "completed.json").read_text())["elapsed_seconds"]
                print(f"Verified resume: {folder.name}", flush=True)
            else:
                folder.mkdir(exist_ok=True)
                provenance.save_json(folder / "manifest.json", request)
                provenance.save_json(folder / "command.json", request["command"])
                state.update(current_job=job, job_started_at=provenance.now())
                print(f"Starting {folder.name} ({len(results)}/{len(jobs)} done)", flush=True)
                start = time.monotonic()
                with (folder / "train.log").open("w", encoding="utf-8") as log:
                    process = subprocess.Popen(request["command"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    state["training_pid"] = process.pid
                    provenance.save_json(output / "queue_status.json", state)
                    returncode = process.wait()
                    process = None
                if returncode:
                    raise RuntimeError(f"{folder.name}: exit={returncode}; inspect train.log")
                sessions = audit_run(folder, request["effective_args"])
                elapsed = time.monotonic() - start
                provenance.save_json(folder / "completed.json", {"fingerprint": request["fingerprint"],
                    "finished_at": provenance.now(), "elapsed_seconds": elapsed, "sessions": sessions})
            results.append({**job, "sessions": sessions, "elapsed_seconds": elapsed})
            state["completed"].append(job["name"])
            state.pop("training_pid", None)
            provenance.save_json(output / "study_results.json", {"smoke": args.smoke,
                "planned_jobs": len(jobs), "complete": len(results) == len(jobs), "runs": results})
            provenance.save_json(output / "queue_status.json", state)
        state.update(state="complete", finished_at=provenance.now())
        state.pop("current_job", None)
        provenance.save_json(output / "queue_status.json", state)
    except BaseException as exc:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=30)
        state.update(state="failed", error=str(exc), finished_at=provenance.now())
        provenance.save_json(output / "queue_status.json", state)
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
