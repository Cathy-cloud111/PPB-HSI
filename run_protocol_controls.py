"""Fresh server controls, audited completion, immutable commands, safe resume.

Default: 30 jobs, not new state-of-the-art baselines. Formal CPU requires opt-in.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import run_capacity_experiments as audit

ROOT = Path(__file__).resolve().parent
VARIANTS = ("er_replay", "ft_sampled", "ft_seen", "lwf_sampled", "lwf_seen")


def build_command(job, output, device, houston=None, pavia=None, smoke=False):
    # Inherit every common effective setting from the formal capacity grid.
    reference = {**job, "variant": "fixed_k5"}
    command = audit.build_command(reference, output, device, houston, pavia, smoke)
    command[2] = str(ROOT / "main_protocol_controls.py")
    flags = dict(zip(command[3::2], command[4::2]))
    variant = job["variant"]
    if variant not in VARIANTS:
        raise ValueError(f"Unknown control {variant}")
    flags.update({"--use_prototypes": "0", "--adaptive_prototypes": "0",
                  "--use_prototype_offsets": "0", "--lambda_old_logit_distill": "0.0",
                  "--incremental_train": "replay" if variant == "er_replay" else "current",
                  "--replay_old_samples_per_class": "100" if variant == "er_replay" else "0",
                  "--old_logit_distill_scope": "all" if variant.startswith("lwf") else "old",
                  "--supervised_class_scope": "seen" if variant.endswith("seen") else "sampled"})
    if variant.startswith("lwf"):
        flags["--lambda_old_logit_distill"] = "1.0"
    return command[:3] + [value for item in flags.items() for value in item]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", choices=audit.DATASETS, default=list(audit.DATASETS))
    p.add_argument("--seeds", nargs="+", type=int, default=list(audit.SEEDS))
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    p.add_argument("--houston-data-dir", type=Path)
    p.add_argument("--pavia-data-dir", type=Path)
    p.add_argument("--output-root", type=Path)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--gpu")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--threads", type=int, default=4)
    return p


def main():
    args = parser().parse_args()
    if args.threads < 1:
        raise ValueError("Positive thread count required")
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    import torch
    import numpy as np
    import scipy
    import main_protocol_controls as entry
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if not args.dry_run:
        if device == "cuda" and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
            raise RuntimeError("One available visible CUDA GPU required; use --gpu")
        if device == "cpu" and not args.smoke and not args.allow_cpu:
            raise RuntimeError("Formal CPU training refused; use server or --smoke")
    output = (args.output_root or ROOT / ("protocol_smoke_outputs" if args.smoke else "protocol_outputs")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    jobs = audit.jobs_for(args.datasets, args.seeds, args.variants)
    if len(jobs) != len({j["name"] for j in jobs}):
        raise ValueError("Duplicate jobs")
    code = {str(p.relative_to(ROOT)): audit.digest(p) for p in (
        ROOT/"main_houston_cls.py", ROOT/"main_protocol_controls.py", Path(__file__),
        ROOT/"run_capacity_experiments.py", ROOT/"datasets/houston_cls.py", ROOT/"datasets/__init__.py")}
    runtime = {"python": platform.python_version(), "torch": str(torch.__version__),
               "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform(),
               "device": device, "threads": args.threads,
               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    data_hashes = {}
    for dataset in args.datasets:
        flags = audit.dataset_args(dataset, args.houston_data_dir, args.pavia_data_dir)
        data_hashes[dataset] = {flags[i]: audit.digest(Path(flags[i+1])) for i in (0, 2, 4)}
    requests = []
    for job in jobs:
        command = build_command(job, output, device, args.houston_data_dir, args.pavia_data_dir, args.smoke)
        effective = vars(entry.get_args_parser().parse_args(command[3:]))
        identity = {"smoke": args.smoke, "protocol": audit.scientific_args(effective),
                    "code_sha256": code, "data_sha256": data_hashes[job["dataset"]], "runtime": runtime}
        requests.append({"job": job, "effective_args": effective, "identity": identity,
                         "fingerprint": audit.fingerprint(identity), "command": command})
    if args.dry_run:
        audit.save_json(output/"plan.json", {"smoke": args.smoke, "total_jobs": len(jobs), "requests": requests})
        print(f"Plan only: {len(jobs)} jobs; no training started.", flush=True)
        return
    lock = output/".queue.lock"
    fd = os.open(lock, os.O_WRONLY|os.O_CREAT|os.O_EXCL)
    with os.fdopen(fd, "w") as handle:
        json.dump({"pid": os.getpid(), "started_at": audit.now()}, handle)
    state = {"started_at": audit.now(), "runner_pid": os.getpid(), "smoke": args.smoke,
             "total_jobs": len(jobs), "completed": [], "state": "running", "runtime": runtime}
    results, process = [], None
    try:
        existing = {}
        for request in requests:
            folder = output/request["job"]["name"]
            if folder.exists() and any(folder.iterdir()):
                existing[folder.name] = audit.require_resume(folder, request)
        audit.save_json(output/"study_plan.json", {"requests": requests, "total_jobs": len(jobs), "smoke": args.smoke})
        for request in requests:
            job = request["job"]
            folder = output/job["name"]
            if job["name"] in existing:
                sessions = existing[job["name"]]
                elapsed = json.loads((folder/"completed.json").read_text())["elapsed_seconds"]
                print(f"Verified resume: {job['name']}", flush=True)
            else:
                folder.mkdir(exist_ok=True)
                audit.save_json(folder/"manifest.json", request)
                audit.save_json(folder/"command.json", request["command"])
                state.update(current_job=job, job_started_at=audit.now())
                print(f"Starting {job['name']} ({len(state['completed'])}/{len(jobs)} done)", flush=True)
                start = time.monotonic()
                with (folder/"train.log").open("w", encoding="utf-8") as log:
                    process = subprocess.Popen(request["command"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    state["training_pid"] = process.pid
                    audit.save_json(output/"queue_status.json", state)
                    returncode = process.wait()
                    process = None
                if returncode:
                    raise RuntimeError(f"{job['name']} exit={returncode}; check train.log")
                elapsed = time.monotonic()-start
                sessions = audit.audit_run(folder, request["effective_args"])
                audit.save_json(folder/"completed.json", {"fingerprint": request["fingerprint"],
                    "finished_at": audit.now(), "elapsed_seconds": elapsed, "sessions": sessions})
            results.append({**job, "elapsed_seconds": elapsed, "sessions": sessions})
            state["completed"].append(job["name"])
            state.pop("training_pid", None)
            audit.save_json(output/"study_results.json", {"smoke": args.smoke,
                "planned_jobs": len(jobs), "complete": len(results)==len(jobs), "runs": results})
            audit.save_json(output/"queue_status.json", state)
        state.update(state="complete", finished_at=audit.now())
        state.pop("current_job", None)
        audit.save_json(output/"queue_status.json", state)
    except BaseException as exc:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=30)
        state.update(state="failed", error=str(exc), finished_at=audit.now())
        audit.save_json(output/"queue_status.json", state)
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
