"""Twelve-run matched persistent-memory ER/Full queue with paired audits."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import run_capacity_experiments as provenance

from persistent_control_audit import audit_run, require_resume, scientific_args, check_pair

ROOT = Path(__file__).resolve().parent
VARIANTS = ("persistent_er", "persistent_full")


def build_command(job, output, device, houston=None, pavia=None, smoke=False, threads=4):
    reference = {**job, "variant": "adaptive_eta003_lse"}
    command = provenance.build_command(reference, output, device, houston, pavia, smoke)
    flags = dict(zip(command[3::2], command[4::2]))
    if job["variant"] == "persistent_er":
        flags.update({"--use_prototypes": "0", "--adaptive_prototypes": "0",
                      "--use_prototype_offsets": "0", "--lambda_old_logit_distill": "0"})
    elif job["variant"] != "persistent_full":
        raise ValueError("Unknown persistent control")
    classes = 15 if job["dataset"] == "Houston2013" else 9
    flags.update({"--persistent_variant": job["variant"],
                  "--memory_budget": str(classes * (4 if smoke else 100)),
                  "--threads": str(threads)})
    command = command[:2] + [str(ROOT / "main_persistent_controls.py")] + [
        value for item in flags.items() for value in item]
    if smoke:
        command.append("--smoke")
    return command


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
    import main_persistent_controls as entry
    torch.set_num_threads(args.threads)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if not args.dry_run and device == "cuda" and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError("Exactly one visible GPU required; use --gpu")
    if not args.dry_run and device == "cpu" and not args.smoke and not args.allow_cpu:
        raise RuntimeError("Formal CPU training refused")
    output = (args.output_root or ROOT / ("persistent_smoke_outputs" if args.smoke else "persistent_outputs")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    jobs = provenance.jobs_for(args.datasets, args.seeds, VARIANTS)
    if len(jobs) != len({j["name"] for j in jobs}):
        raise ValueError("Duplicate jobs")
    code = {str(p.relative_to(ROOT)): provenance.digest(p) for p in (
        ROOT / "main_persistent_controls.py", ROOT / "main_icarl_hsi.py",
        ROOT / "persistent_control_audit.py", Path(__file__), ROOT / "main_houston_cls.py",
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
            "interpretation": "Matched ER/Full with identical random-priority persistent images, fixed total memory and training RNG schedule; not identical iCaRL selection or whole-scene memory."}
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
        pairs = []
        for dataset in args.datasets:
            for seed in args.seeds:
                pair_requests = [next(r for r in requests if r["job"]["dataset"] == dataset
                    and r["job"]["seed"] == seed and r["job"]["variant"] == variant) for variant in VARIANTS]
                pair_sessions = [next(r["sessions"] for r in results if r["name"] == request["job"]["name"])
                                 for request in pair_requests]
                pairs.append(check_pair(*pair_requests, *pair_sessions))
        provenance.save_json(output / "pair_checks.json", {"smoke": args.smoke, "pairs": pairs})
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
