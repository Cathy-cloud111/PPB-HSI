"""Prepare and run missing controlled CIL baselines for the ICASSP paper.

The default queue contains only baselines that are not already available in
``verified_outputs``:

* ``lwf``: current-class training with old-logit distillation on every current
  sample and no replay memory;
* ``cumulative``: access to all previously seen training samples, reported as
  an upper reference rather than as a memory-constrained CIL method.

The existing ``head_only_replay`` runs are reused as the ER-style comparison.
iCaRL is deliberately not included here: a faithful implementation additionally
requires persistent herding exemplars and nearest-mean-of-exemplars inference.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "comparison_outputs"
SEEDS = (42, 123, 3407)
DATASETS = ("Houston2013", "PaviaU")
DEFAULT_VARIANTS = ("lwf", "cumulative")


def now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def dataset_args(dataset, houston_data_dir=None, pavia_data_dir=None):
    if dataset == "Houston2013":
        data = (Path(houston_data_dir) if houston_data_dir else
                ROOT / "data" / "Houston2013")
        return [
            "--hsi_file", str(data / "HSI.mat"),
            "--train_label_file", str(data / "TRLabel.mat"),
            "--test_label_file", str(data / "TSLabel.mat"),
            "--hsi_key", "HSI",
            "--sessions", "1-9,10-12,13-15",
            "--num_classes", "15",
        ]

    data = Path(pavia_data_dir) if pavia_data_dir else ROOT / "data" / "PaviaU"
    return [
        "--hsi_file", str(data / "PaviaU.mat"),
        "--train_label_file", str(data / "PaviaU_train_spatial.mat"),
        "--test_label_file", str(data / "PaviaU_test_spatial.mat"),
        "--hsi_key", "paviaU",
        "--hsi_label_key", "paviaU_gt",
        "--sessions", "1-5,6-7,8-9",
        "--num_classes", "9",
    ]


def variant_args(variant):
    common = [
        "--use_prototypes", "0",
        "--adaptive_prototypes", "0",
        "--use_prototype_offsets", "0",
    ]
    if variant == "lwf":
        return common + [
            "--incremental_train", "current",
            "--replay_old_samples_per_class", "0",
            "--lambda_old_logit_distill", "1.0",
            "--old_logit_distill_temperature", "2.0",
            "--old_logit_distill_scope", "all",
        ]
    if variant == "cumulative":
        return common + [
            "--incremental_train", "cumulative",
            "--replay_old_samples_per_class", "0",
            "--lambda_old_logit_distill", "0.0",
            "--old_logit_distill_scope", "old",
        ]
    raise ValueError(f"Unknown variant: {variant}")


def build_command(job, python, device, houston_data_dir=None, pavia_data_dir=None):
    folder = OUT / job["name"]
    command = [python, "-u", str(ROOT / "main_houston_cls.py")]
    command += dataset_args(job["dataset"], houston_data_dir, pavia_data_dir)
    command += [
        "--model", "pdp",
        "--hsi_patch_size", "15",
        "--epochs", "80",
        "--batch_size", "128",
        "--num_workers", "0",
        "--device", device,
        "--num_prototypes_per_class", "5",
        "--prototype_pooling", "logsumexp",
        "--freeze_backbone_after_base", "1",
        "--freeze_shared_prompts_after_base", "1",
        "--seed", str(job["seed"]),
        "--output_dir", str(folder),
    ]
    command += variant_args(job["variant"])
    return command


def summarize(jobs):
    rows = []
    for job in jobs:
        metrics = OUT / job["name"] / "metrics.csv"
        if not metrics.exists():
            continue
        with metrics.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append({
                    "dataset": job["dataset"],
                    "variant": job["variant"],
                    "seed": job["seed"],
                    **{key: row[key] for key in (
                        "session", "oa", "aa", "kappa", "current_oa",
                        "apd_base_forgetting",
                    )},
                })
    if not rows:
        return
    with (OUT / "all_runs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--houston-data-dir", type=Path, default=None)
    parser.add_argument("--pavia-data-dir", type=Path, default=None)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["lwf", "cumulative"],
        default=list(DEFAULT_VARIANTS),
    )
    args = parser.parse_args()
    for dataset in DATASETS:
        data_flags = dataset_args(dataset, args.houston_data_dir, args.pavia_data_dir)
        for flag in ("--hsi_file", "--train_label_file", "--test_label_file"):
            path = Path(data_flags[data_flags.index(flag) + 1])
            if not path.is_file():
                raise FileNotFoundError(f"{dataset}: missing {flag}: {path}")
    OUT.mkdir(exist_ok=True)

    import scipy  # noqa: F401 -- dependency check before queue start
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    jobs = [
        {
            "name": f"{dataset}_{variant}_seed{seed}",
            "dataset": dataset,
            "variant": variant,
            "seed": seed,
        }
        for dataset in DATASETS
        for seed in SEEDS
        for variant in args.variants
    ]
    state = {
        "started_at": now(),
        "runner_pid": os.getpid(),
        "python": sys.executable,
        "torch_version": torch.__version__,
        "device": device,
        "total_jobs": len(jobs),
        "completed": [],
        "state": "dry_run" if args.dry_run else "running",
    }

    for job in jobs:
        folder = OUT / job["name"]
        folder.mkdir(exist_ok=True)
        command = build_command(
            job, sys.executable, device, args.houston_data_dir, args.pavia_data_dir
        )
        (folder / "command.json").write_text(
            json.dumps(command, indent=2), encoding="utf-8"
        )
        if (folder / "completed.json").exists():
            state["completed"].append(job["name"])
            continue
        if args.dry_run:
            continue

        state.update(current_job=job, job_started_at=now())
        save_json(OUT / "queue_status.json", state)
        with (folder / "train.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
            )
            state["training_pid"] = process.pid
            save_json(OUT / "queue_status.json", state)
            return_code = process.wait()
        if return_code:
            state.update(state="failed", exit_code=return_code, finished_at=now())
            save_json(OUT / "queue_status.json", state)
            summarize(jobs)
            raise SystemExit(return_code)

        with (folder / "metrics.csv").open(
            encoding="utf-8-sig", newline=""
        ) as handle:
            sessions = {int(row["session"]) for row in csv.DictReader(handle)}
        if sessions != {1, 2, 3}:
            state.update(
                state="failed",
                error=f"Incomplete session metrics: {sorted(sessions)}",
                finished_at=now(),
            )
            save_json(OUT / "queue_status.json", state)
            raise SystemExit(1)

        save_json(
            folder / "completed.json",
            {"finished_at": now(), "exit_code": return_code},
        )
        state["completed"].append(job["name"])
        summarize(jobs)
        save_json(OUT / "queue_status.json", state)

    state.update(
        state="dry_run_complete" if args.dry_run else "complete",
        finished_at=now(),
    )
    save_json(OUT / "queue_status.json", state)


if __name__ == "__main__":
    main()
