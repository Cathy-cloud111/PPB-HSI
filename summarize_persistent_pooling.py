"""Re-audit the main-protocol LSE/LME pair and report paired differences."""
import argparse
import json
from pathlib import Path
import statistics

import run_capacity_experiments as provenance
import run_persistent_ablation as runner
from persistent_control_audit import check_matched_group, require_resume


def stats(values):
    return {"mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values),
            "values": list(values), "seeds": list(provenance.SEEDS),
            "n": len(values)}


def summarize(root, houston=None, pavia=None):
    variants = runner.POOLING_VARIANTS
    study = json.loads((root / "study_results.json").read_text(encoding="utf-8"))
    plan = json.loads((root / "study_plan.json").read_text(encoding="utf-8"))
    state = json.loads((root / "queue_status.json").read_text(encoding="utf-8"))
    stored_checks = json.loads((root / "group_checks.json").read_text(encoding="utf-8"))
    declared = provenance.jobs_for(provenance.DATASETS, provenance.SEEDS, variants)
    names = {job["name"] for job in declared}
    total = len(declared)
    if total != 12 or plan.get("variants") != list(variants):
        raise ValueError("Expected the predeclared twelve-run LSE/LME study")
    if any(item.get("smoke") for item in (study, plan, state, stored_checks)):
        raise ValueError("Smoke outputs are not research evidence")
    if not study.get("complete") or state.get("state") != "complete":
        raise ValueError("Require a complete formal pooling queue")
    if (plan.get("total_jobs") != total or study.get("planned_jobs") != total or
            len(plan.get("requests", [])) != total or
            len(study.get("runs", [])) != total or
            {r["job"]["name"] for r in plan["requests"]} != names or
            {r["name"] for r in study["runs"]} != names or
            set(state.get("completed", [])) != names):
        raise ValueError("Pooling study is incomplete or duplicated")

    runs, requests, data_cache = {}, {}, {}
    for index, original in enumerate(plan["requests"], 1):
        request = {**original, "effective_args": dict(original["effective_args"])}
        job = request["job"]
        if request["fingerprint"] != provenance.fingerprint(request["identity"]):
            raise ValueError("Invalid scientific fingerprint")
        print(f"Auditing {index}/{total}: {job['name']}", flush=True)
        if houston is not None or pavia is not None:
            flags = provenance.dataset_args(job["dataset"], houston, pavia)
            for position in (0, 2, 4):
                request["effective_args"][flags[position][2:]] = flags[position + 1]
        for flag, expected in request["identity"]["data_sha256"].items():
            path = Path(request["effective_args"][flag[2:]])
            if path not in data_cache:
                data_cache[path] = provenance.digest(path)
            if data_cache[path] != expected:
                raise ValueError("Audit data differ from locked training contents")
        sessions = require_resume(root / job["name"], request)
        saved = next(r for r in study["runs"] if r["name"] == job["name"])
        if saved["sessions"] != sessions:
            raise ValueError("Queue aggregate differs from audited artifacts")
        key = (job["dataset"], job["seed"], job["variant"])
        if key in runs:
            raise ValueError("Duplicate audited run")
        runs[key], requests[key] = sessions, request

    checks = []
    for dataset in provenance.DATASETS:
        for seed in provenance.SEEDS:
            keys = [(dataset, seed, variant) for variant in variants]
            checks.append(check_matched_group(
                [requests[key] for key in keys], [runs[key] for key in keys]))
    if stored_checks.get("groups") != checks:
        raise ValueError("Stored pair checks differ from the re-audit")

    groups, differences = [], []
    for dataset in provenance.DATASETS:
        for variant in variants:
            metrics = {metric: stats([runs[dataset, seed, variant][-1][metric]
                                      for seed in provenance.SEEDS])
                       for metric in provenance.METRICS}
            groups.append({"dataset": dataset, "variant": variant,
                           "metrics": metrics})
            print(f"{dataset:11s} {variant:20s} "
                  f"OA={100 * metrics['oa']['mean']:.2f}+/-"
                  f"{100 * metrics['oa']['sample_sd']:.2f}%", flush=True)
        metrics = {metric: stats([
            100 * (runs[dataset, seed, "persistent_pool_lme"][-1][metric] -
                   runs[dataset, seed, "persistent_pool_lse"][-1][metric])
            for seed in provenance.SEEDS]) for metric in provenance.METRICS}
        differences.append({"dataset": dataset,
                            "positive": "persistent_pool_lme",
                            "negative": "persistent_pool_lse",
                            "units": "percentage points", "metrics": metrics})
        print(f"{dataset:11s} LME-LSE: OA={metrics['oa']['mean']:+.4f}pp; "
              f"Base={metrics['base_oa']['mean']:+.4f}pp; "
              f"Current={metrics['current_oa']['mean']:+.4f}pp; "
              f"OA seeds={metrics['oa']['values']}", flush=True)

    report = {"audited_runs": total, "groups": groups,
              "paired_differences": differences, "matched_pair_checks": checks,
              "runs": [{"dataset": dataset, "seed": seed, "variant": variant,
                         "sessions": sessions}
                       for (dataset, seed, variant), sessions in sorted(runs.items())],
              "limits": "Paired LSE/LME comparison under identical fixed-total-memory "
                        "inputs, persistent images, initialization, epoch order and RNG. "
                        "Three seeds do not establish statistical significance."}
    provenance.save_json(root / "persistent_pooling_report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path,
                        default=runner.ROOT / "persistent_pooling_outputs")
    parser.add_argument("--houston-data-dir", type=Path)
    parser.add_argument("--pavia-data-dir", type=Path)
    args = parser.parse_args()
    summarize(args.output_root, args.houston_data_dir, args.pavia_data_dir)
