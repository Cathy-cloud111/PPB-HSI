"""Re-audit the six-run strict-spatial Houston ER/Full experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import run_capacity_experiments as provenance
import run_persistent_controls as runner


DATASET = "Houston2013"


def stats(values):
    values = list(values)
    return {
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values),
        "values": values,
        "seeds": list(provenance.SEEDS),
        "n": len(values),
    }


def summarize(root: Path, houston: Path):
    study = json.loads((root / "study_results.json").read_text(encoding="utf-8"))
    plan = json.loads((root / "study_plan.json").read_text(encoding="utf-8"))
    state = json.loads((root / "queue_status.json").read_text(encoding="utf-8"))
    split = json.loads((houston / "Houston2013_spatial_split.json").read_text(encoding="utf-8"))
    declared = provenance.jobs_for([DATASET], provenance.SEEDS, runner.VARIANTS)
    names = {job["name"] for job in declared}
    if not split.get("spatially_disjoint_patches") or split.get("train_test_patch_overlap_count") != 0:
        raise ValueError("Houston split is not patch-disjoint")
    if any(value["train"] < 1 or value["test"] < 1 for value in split["classes"].values()):
        raise ValueError("Every class must retain train and test centres")
    if study["smoke"] or plan["smoke"] or state["smoke"]:
        raise ValueError("Smoke output cannot be used as research evidence")
    if not study["complete"] or state["state"] != "complete":
        raise ValueError("Formal queue is incomplete")
    if not (plan["total_jobs"] == study["planned_jobs"] == len(declared) == 6):
        raise ValueError("Expected six declared formal runs")
    if ({run["name"] for run in study["runs"]} != names or
            {request["job"]["name"] for request in plan["requests"]} != names or
            set(state["completed"]) != names):
        raise ValueError("Queue contents differ from the declared study")

    runs, requests, data_cache = {}, {}, {}
    for index, original in enumerate(plan["requests"], 1):
        request = {**original, "effective_args": dict(original["effective_args"])}
        job = request["job"]
        if request["fingerprint"] != provenance.fingerprint(request["identity"]):
            raise ValueError("Invalid request fingerprint")
        if request["identity"]["protocol"] != runner.scientific_args(request["effective_args"]):
            raise ValueError("Scientific protocol differs from its locked identity")
        flags = provenance.dataset_args(DATASET, houston, None)
        for position in (0, 2, 4):
            request["effective_args"][flags[position][2:]] = flags[position + 1]
        for flag, expected in request["identity"]["data_sha256"].items():
            path = Path(request["effective_args"][flag[2:]])
            if path not in data_cache:
                data_cache[path] = provenance.digest(path)
            if data_cache[path] != expected:
                raise ValueError(f"Training input changed after the run: {path}")
        print(f"Auditing {index}/6: {job['name']}", flush=True)
        sessions = runner.require_resume(root / job["name"], request)
        saved = next(run for run in study["runs"] if run["name"] == job["name"])
        if saved["sessions"] != sessions:
            raise ValueError("Queue aggregate differs from audited artifacts")
        key = (job["seed"], job["variant"])
        runs[key] = sessions
        requests[key] = request

    pairs = []
    for seed in provenance.SEEDS:
        er, full = [(seed, variant) for variant in runner.VARIANTS]
        pairs.append(runner.check_pair(
            requests[er], requests[full], runs[er], runs[full]))
    stored_pairs = json.loads((root / "pair_checks.json").read_text(encoding="utf-8"))
    if stored_pairs["smoke"] or stored_pairs["pairs"] != pairs:
        raise ValueError("Stored paired checks differ from the independent re-audit")

    groups = []
    for variant in runner.VARIANTS:
        metrics = {
            metric: stats(runs[seed, variant][-1][metric] for seed in provenance.SEEDS)
            for metric in provenance.METRICS
        }
        metrics["average_incremental_oa"] = stats(
            statistics.mean(session["oa"] for session in runs[seed, variant])
            for seed in provenance.SEEDS
        )
        groups.append({"dataset": DATASET, "variant": variant, "metrics": metrics})
        print(
            f"{variant}: OA={100 * metrics['oa']['mean']:.2f}"
            f"+/-{100 * metrics['oa']['sample_sd']:.2f}%",
            flush=True,
        )

    contrast = {
        metric: stats(
            100 * (runs[seed, "persistent_full"][-1][metric] -
                   runs[seed, "persistent_er"][-1][metric])
            for seed in provenance.SEEDS
        )
        for metric in provenance.METRICS
    }
    print(
        f"Full-ER: OA={contrast['oa']['mean']:+.4f}pp; "
        f"Base={contrast['base_oa']['mean']:+.4f}pp; "
        f"Current={contrast['current_oa']['mean']:+.4f}pp",
        flush=True,
    )
    report = {
        "dataset": DATASET,
        "protocol": "fresh 10% class-wise spatial-region split with a 14-pixel guard for 15x15 patches",
        "split": split,
        "audited_runs": 6,
        "groups": groups,
        "paired_difference": {
            "direction": "persistent_full minus persistent_er",
            "units": "percentage points",
            "metrics": contrast,
        },
        "pair_checks": pairs,
        "runs": [
            {"seed": seed, "variant": variant, "sessions": sessions}
            for (seed, variant), sessions in runs.items()
        ],
        "limits": (
            "Robustness result under a fresh guarded spatial split. It is not numerically "
            "pooled with the official Houston split and does not establish significance."
        ),
    }
    provenance.save_json(root / "spatial_protocol_report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=runner.ROOT / "strict_spatial_outputs")
    parser.add_argument("--houston-data-dir", required=True, type=Path)
    args = parser.parse_args()
    summarize(args.output_root.resolve(), args.houston_data_dir.resolve())
