"""Re-audit all completed iCaRL reference runs; no automatic paper edits."""
import argparse
import json
from pathlib import Path
import statistics
import run_icarl_hsi as runner
import run_capacity_experiments as provenance


def summarize(root, houston=None, pavia=None):
    study = json.loads((root / "study_results.json").read_text())
    plan = json.loads((root / "study_plan.json").read_text())
    if study["smoke"] or plan["smoke"] or not study["complete"]:
        raise ValueError("Only a complete formal study can be summarized for the paper")
    if len(study["runs"]) != plan["total_jobs"] or len(study["runs"]) != study["planned_jobs"]:
        raise ValueError("Incomplete run count")
    if {r["name"] for r in study["runs"]} != {r["job"]["name"] for r in plan["requests"]}:
        raise ValueError("Plan/result job mismatch")
    audited, data_hash_cache = [], {}
    for i, request in enumerate(plan["requests"], 1):
        job = request["job"]
        print(f"Auditing {i}/{plan['total_jobs']}: {job['name']}", flush=True)
        # Local transfers can relocate files, but must retain identical contents.
        request = {**request, "effective_args": dict(request["effective_args"])}
        if houston is not None or pavia is not None:
            flags = provenance.dataset_args(job["dataset"], houston, pavia)
            for index in (0, 2, 4):
                request["effective_args"][flags[index][2:]] = flags[index + 1]
        for flag, expected_hash in request["identity"]["data_sha256"].items():
            path = Path(request["effective_args"][flag[2:]])
            if path not in data_hash_cache:
                data_hash_cache[path] = provenance.digest(path)
            if data_hash_cache[path] != expected_hash:
                raise ValueError(f"Data contents differ from locked manifest: {path}")
        sessions = runner.require_resume(root / job["name"], request)
        original = next(r for r in study["runs"] if r["name"] == job["name"])
        if sessions != original["sessions"]:
            raise ValueError("Queue aggregate differs from recomputed results")
        audited.append({**job, "sessions": sessions})
    groups = []
    for dataset in dict.fromkeys(r["dataset"] for r in audited):
        runs = [r for r in audited if r["dataset"] == dataset]
        seeds = [r["seed"] for r in runs]
        if sorted(seeds) != sorted(provenance.SEEDS):
            raise ValueError("Formal paper report requires all three declared seeds")
        values = {}
        for key in provenance.METRICS:
            samples = [r["sessions"][-1][key] for r in runs]
            values[key] = {"mean": statistics.mean(samples), "sample_sd": statistics.stdev(samples),
                           "seeds": seeds, "values": samples}
        incremental = [statistics.mean(s["oa"] for s in r["sessions"]) for r in runs]
        values["average_incremental_oa"] = {"mean": statistics.mean(incremental),
            "sample_sd": statistics.stdev(incremental), "seeds": seeds, "values": incremental}
        groups.append({"dataset": dataset, "variant": "icarl_hsi", "metrics": values})
        print(f"{dataset}: OA={values['oa']['mean'] * 100:.2f}+/-{values['oa']['sample_sd'] * 100:.2f}%", flush=True)
    report = {"audited_runs": len(audited), "groups": groups, "runs": audited,
              "interpretation": "Independent HSI adaptation, not original published scores or memory-matched superiority evidence."}
    provenance.save_json(root / "icarl_report.json", report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, default=runner.ROOT / "icarl_outputs")
    p.add_argument("--houston-data-dir", type=Path)
    p.add_argument("--pavia-data-dir", type=Path)
    args = p.parse_args()
    summarize(args.output_root, args.houston_data_dir, args.pavia_data_dir)
