"""Re-audit matched persistent-memory ER/Full runs, all seeds and contrasts."""
import argparse
import json
from pathlib import Path
import statistics

import run_persistent_controls as runner
import run_capacity_experiments as provenance


def stats(values):
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values),
            "values": list(values), "seeds": list(provenance.SEEDS), "n": len(values)}


def summarize(root, houston=None, pavia=None):
    study = json.loads((root / "study_results.json").read_text())
    plan = json.loads((root / "study_plan.json").read_text())
    state = json.loads((root / "queue_status.json").read_text())
    declared = provenance.jobs_for(provenance.DATASETS, provenance.SEEDS, runner.VARIANTS)
    if study["smoke"] or plan["smoke"] or state["smoke"] or not study["complete"] or state["state"] != "complete":
        raise ValueError("Require complete formal queue; smoke is not research evidence")
    names = {job["name"] for job in declared}
    if plan["total_jobs"] != 12 or study["planned_jobs"] != 12 or len(study["runs"]) != 12 or (
            len(plan["requests"]) != 12 or {r["name"] for r in study["runs"]} != names or
            {r["job"]["name"] for r in plan["requests"]} != names or set(state["completed"]) != names):
        raise ValueError("Require all declared twelve formal runs")
    runs, requests, data_cache = {}, {}, {}
    for index, original in enumerate(plan["requests"], 1):
        request = {**original, "effective_args": dict(original["effective_args"])}
        job = request["job"]
        if request["fingerprint"] != provenance.fingerprint(request["identity"]) or (
                request["identity"]["protocol"] != runner.scientific_args(request["effective_args"])):
            raise ValueError("Invalid scientific identity")
        print(f"Auditing {index}/12: {job['name']}", flush=True)
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
        sessions = runner.require_resume(root / job["name"], request)
        saved = next(r for r in study["runs"] if r["name"] == job["name"])
        if saved["sessions"] != sessions:
            raise ValueError("Queue aggregate differs from artifacts")
        key = (job["dataset"], job["seed"], job["variant"])
        runs[key] = sessions
        requests[key] = request
    pairs, groups, contrasts = [], [], []
    for dataset in provenance.DATASETS:
        for seed in provenance.SEEDS:
            er, full = [(dataset, seed, variant) for variant in runner.VARIANTS]
            pairs.append(runner.check_pair(requests[er], requests[full], runs[er], runs[full]))
        for variant in runner.VARIANTS:
            metrics = {metric: stats([runs[dataset, seed, variant][-1][metric] for seed in provenance.SEEDS])
                       for metric in provenance.METRICS}
            metrics["average_incremental_oa"] = stats([
                statistics.mean(s["oa"] for s in runs[dataset, seed, variant]) for seed in provenance.SEEDS])
            groups.append({"dataset": dataset, "variant": variant, "metrics": metrics})
            print(f"{dataset} {variant}: OA={100*metrics['oa']['mean']:.2f}+/-{100*metrics['oa']['sample_sd']:.2f}%", flush=True)
        contrast = {metric: stats([100 * (runs[dataset, seed, "persistent_full"][-1][metric] -
            runs[dataset, seed, "persistent_er"][-1][metric]) for seed in provenance.SEEDS])
            for metric in provenance.METRICS}
        contrasts.append({"dataset": dataset, "direction": "persistent_full minus persistent_er",
                          "units": "percentage points", "metrics": contrast})
        print(f"{dataset} Full-ER: OA={contrast['oa']['mean']:+.4f}pp; seeds={contrast['oa']['values']}", flush=True)
    queue_pairs = json.loads((root / "pair_checks.json").read_text())
    if queue_pairs["smoke"] or queue_pairs["pairs"] != pairs:
        raise ValueError("Stored queue pair checks differ from re-audit")
    report = {"audited_runs": 12, "groups": groups, "paired_differences": contrasts, "pair_checks": pairs,
              "runs": [{"dataset": d, "seed": s, "variant": v, "sessions": stages}
                       for (d, s, v), stages in runs.items()],
              "limits": "Whole-bundle conditional ER/Full difference under identical random-priority image memory, "
                        "initialization, inputs, epoch order and declared RNG seeds. Not individual K/IPOC effects, "
                        "not identical iCaRL exemplar selection, not a significance or SOTA claim; GPU arithmetic "
                        "is not guaranteed bit-identical. Calibration adds method-specific optimization."}
    provenance.save_json(root / "persistent_report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=runner.ROOT / "persistent_outputs")
    parser.add_argument("--houston-data-dir", type=Path)
    parser.add_argument("--pavia-data-dir", type=Path)
    args = parser.parse_args()
    summarize(args.output_root, args.houston_data_dir, args.pavia_data_dir)
