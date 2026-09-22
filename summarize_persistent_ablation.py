"""Re-audit the locked persistent-memory component ablation and contrasts."""
import argparse
import json
from pathlib import Path
import statistics

import run_capacity_experiments as provenance
import run_persistent_ablation as runner
from persistent_control_audit import PAIR_FIELDS, check_matched_group, require_resume

REFERENCE_VARIANTS = ("persistent_er", "persistent_full")
ALL_VARIANTS = REFERENCE_VARIANTS + runner.VARIANTS
CONTRASTS = (
    ("Full minus ER", "persistent_full", "persistent_er"),
    ("KD alone minus ER", "persistent_kd", "persistent_er"),
    ("Multi-prototype alone minus ER", "persistent_multiproto", "persistent_er"),
    ("KD given multi-prototype", "persistent_multiproto_kd", "persistent_multiproto"),
    ("IPOC given multi-prototype", "persistent_multiproto_ipoc", "persistent_multiproto"),
    ("KD given multi-prototype and IPOC", "persistent_full", "persistent_multiproto_ipoc"),
    ("IPOC given multi-prototype and KD", "persistent_full", "persistent_multiproto_kd"),
    ("Multi-prototype given KD", "persistent_multiproto_kd", "persistent_kd"),
    ("Adaptive minus fixed capacity in Full", "persistent_full", "persistent_fixed_full"),
)


def stats(values):
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values),
            "values": list(values), "seeds": list(provenance.SEEDS), "n": len(values)}


def load_reference(path):
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("audited_runs") != 12:
        raise ValueError("Reference report must contain twelve audited ER/Full runs")
    runs = {}
    for run in report.get("runs", []):
        key = (run["dataset"], run["seed"], run["variant"])
        if run["variant"] not in REFERENCE_VARIANTS or key in runs or len(run["sessions"]) != 3:
            raise ValueError("Reference report has an unexpected or duplicate run")
        for row in run["sessions"]:
            if any(field not in row for field in PAIR_FIELDS):
                raise ValueError("Reference report lacks matched-protocol audit fields")
        runs[key] = run["sessions"]
    expected = {(dataset, seed, variant) for dataset in provenance.DATASETS
                for seed in provenance.SEEDS for variant in REFERENCE_VARIANTS}
    if set(runs) != expected:
        raise ValueError("Reference report is incomplete")
    return runs


def assert_reference_match(reference, candidate):
    if len(reference) != len(candidate) or len(reference) != 3:
        raise ValueError("Reference/candidate session count differs")
    for left, right in zip(reference, candidate):
        if left["session"] != right["session"] or any(
                left[field] != right[field] for field in PAIR_FIELDS):
            raise ValueError("New run does not match the audited ER/Full inputs, memory, initialization, order or RNG")


def summarize(root, reference_report, houston=None, pavia=None):
    study = json.loads((root / "study_results.json").read_text(encoding="utf-8"))
    plan = json.loads((root / "study_plan.json").read_text(encoding="utf-8"))
    state = json.loads((root / "queue_status.json").read_text(encoding="utf-8"))
    stored_checks = json.loads((root / "group_checks.json").read_text(encoding="utf-8"))
    declared = provenance.jobs_for(provenance.DATASETS, provenance.SEEDS, runner.VARIANTS)
    expected_names = {job["name"] for job in declared}
    expected_total = len(provenance.DATASETS) * len(provenance.SEEDS) * len(runner.VARIANTS)
    if expected_total != 30:
        raise ValueError("Ablation definition must contain exactly thirty missing runs")
    if study.get("smoke") or plan.get("smoke") or state.get("smoke") or stored_checks.get("smoke"):
        raise ValueError("Smoke outputs are not research evidence")
    if not study.get("complete") or state.get("state") != "complete":
        raise ValueError("Require a complete formal ablation queue")
    if (plan.get("total_jobs") != expected_total or study.get("planned_jobs") != expected_total or
            len(plan.get("requests", [])) != expected_total or len(study.get("runs", [])) != expected_total or
            {request["job"]["name"] for request in plan["requests"]} != expected_names or
            {run["name"] for run in study["runs"]} != expected_names or
            set(state.get("completed", [])) != expected_names):
        raise ValueError("Require all thirty declared formal runs exactly once")

    runs, requests, data_cache = {}, {}, {}
    for index, original in enumerate(plan["requests"], 1):
        request = {**original, "effective_args": dict(original["effective_args"])}
        job = request["job"]
        if request["fingerprint"] != provenance.fingerprint(request["identity"]) or (
                request["identity"]["protocol"] != runner.scientific_args(request["effective_args"])):
            raise ValueError("Invalid scientific identity")
        print(f"Auditing {index}/{expected_total}: {job['name']}", flush=True)
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
        saved = next(run for run in study["runs"] if run["name"] == job["name"])
        if saved["sessions"] != sessions:
            raise ValueError("Queue aggregate differs from audited artifacts")
        key = (job["dataset"], job["seed"], job["variant"])
        if key in runs:
            raise ValueError("Duplicate audited run")
        runs[key], requests[key] = sessions, request

    reference = load_reference(reference_report)
    group_checks, cross_checks = [], []
    for dataset in provenance.DATASETS:
        for seed in provenance.SEEDS:
            keys = [(dataset, seed, variant) for variant in runner.VARIANTS]
            group_checks.append(check_matched_group(
                [requests[key] for key in keys], [runs[key] for key in keys]))
            for variant in runner.VARIANTS:
                assert_reference_match(reference[dataset, seed, "persistent_er"],
                                       runs[dataset, seed, variant])
                cross_checks.append({"dataset": dataset, "seed": seed,
                                     "variant": variant, "reference": "persistent_er",
                                     "passed": True, "checked_fields": list(PAIR_FIELDS)})
    if stored_checks.get("groups") != group_checks:
        raise ValueError("Stored matched-group checks differ from the re-audit")
    runs.update(reference)

    groups = []
    for dataset in provenance.DATASETS:
        for variant in ALL_VARIANTS:
            metrics = {metric: stats([runs[dataset, seed, variant][-1][metric]
                                      for seed in provenance.SEEDS])
                       for metric in provenance.METRICS}
            metrics["average_incremental_oa"] = stats([
                statistics.mean(row["oa"] for row in runs[dataset, seed, variant])
                for seed in provenance.SEEDS])
            groups.append({"dataset": dataset, "variant": variant, "metrics": metrics})
            print(f"{dataset:11s} {variant:28s} "
                  f"OA={100 * metrics['oa']['mean']:.2f}+/-{100 * metrics['oa']['sample_sd']:.2f}%",
                  flush=True)

    contrasts = []
    for dataset in provenance.DATASETS:
        for label, positive, negative in CONTRASTS:
            metrics = {metric: stats([100 * (runs[dataset, seed, positive][-1][metric] -
                                              runs[dataset, seed, negative][-1][metric])
                                      for seed in provenance.SEEDS])
                       for metric in provenance.METRICS}
            contrasts.append({"dataset": dataset, "label": label,
                              "positive": positive, "negative": negative,
                              "units": "percentage points", "metrics": metrics})
            print(f"{dataset:11s} {label}: OA={metrics['oa']['mean']:+.4f}pp; "
                  f"seeds={metrics['oa']['values']}", flush=True)

    report = {
        "audited_new_runs": expected_total,
        "audited_reference_runs": 12,
        "groups": groups,
        "paired_differences": contrasts,
        "matched_group_checks": group_checks,
        "reference_cross_checks": cross_checks,
        "runs": [{"dataset": dataset, "seed": seed, "variant": variant,
                  "sessions": sessions}
                 for (dataset, seed, variant), sessions in sorted(runs.items())],
        "limits": "Paired component contrasts under identical fixed-budget random-priority image memory, "
                  "training inputs, initialization, epoch order and declared RNG seeds. Three seeds do not "
                  "establish statistical significance, and effects need not be additive."
    }
    provenance.save_json(root / "persistent_ablation_report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path,
                        default=runner.ROOT / "persistent_ablation_outputs")
    parser.add_argument("--reference-report", type=Path,
                        default=runner.ROOT / "persistent_outputs" / "persistent_report.json")
    parser.add_argument("--houston-data-dir", type=Path)
    parser.add_argument("--pavia-data-dir", type=Path)
    args = parser.parse_args()
    summarize(args.output_root, args.reference_report,
              args.houston_data_dir, args.pavia_data_dir)
