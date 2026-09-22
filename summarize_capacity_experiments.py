"""Summarize the COMPLETE formal grid, rechecking every saved run first.

No test-set threshold selection, no smoke results, no mixing with old tables.
Optional plots are analysis artifacts, not automatically inserted in the paper.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import run_capacity_experiments as study


def mean_std(values):
    array = np.asarray(values, dtype=float)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=1)), "n": len(array)}


def summarize(output):
    saved = json.loads((output / "study_results.json").read_text(encoding="utf-8"))
    expected = study.jobs_for(study.DATASETS, study.SEEDS, study.VARIANTS)
    expected_names = {j["name"] for j in expected}
    if saved["smoke"] or not saved["complete"] or saved["planned_jobs"] != 54:
        raise ValueError("Paper evidence requires all 54 formal jobs; smoke/partial study refused")
    if len(saved["runs"]) != 54 or {r["name"] for r in saved["runs"]} != expected_names:
        raise ValueError("Formal study does not match the predeclared dataset/seed/configuration grid")
    audited = []
    for job in expected:
        folder = output / job["name"]
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        args = manifest["effective_args"]
        if manifest["job"] != job or manifest["identity"]["smoke"] or args["epochs"] != 80 or (
                args["prototype_offset_epochs"] != 10 or args["max_train_samples_per_class"] or
                args["max_test_samples_per_class"]):
            raise ValueError(f"Not a formal uncapped protocol: {folder}")
        if manifest["fingerprint"] != study.fingerprint(manifest["identity"]):
            raise ValueError(f"Manifest identity mismatch: {folder}")
        command_args = study.build_command(job, output, args["device"])
        import main_houston_cls as engine
        declared = vars(engine.get_args_parser().parse_args(command_args[3:]))
        if study.scientific_args(args) != study.scientific_args(declared):
            raise ValueError(f"Not a predeclared configuration: {folder}")
        sessions = study.require_resume(folder, manifest)
        source = next(r for r in saved["runs"] if r["name"] == job["name"])
        if source["sessions"] != sessions:
            raise ValueError(f"Aggregate results mismatch: {folder}")
        audited.append({**job, "sessions": sessions})
    # Comparing capacities only makes sense with the same engine, data and runtime.
    manifests = [json.loads((output / j["name"] / "manifest.json").read_text()) for j in expected]
    for dataset in study.DATASETS:
        identities = [m["identity"] for m in manifests if m["job"]["dataset"] == dataset]
        for field in ("code_sha256", "data_sha256", "runtime"):
            if any(i[field] != identities[0][field] for i in identities):
                raise ValueError(f"Mixed {field} across {dataset} runs")
    groups, paired = [], []
    metrics = (*study.METRICS, "active_k_mean", "active_prototypes", "allocated_prototype_slots")
    for dataset in study.DATASETS:
        for variant in study.VARIANTS:
            for session in (1, 2, 3):
                rows = [r["sessions"][session - 1] for r in audited
                        if r["dataset"] == dataset and r["variant"] == variant]
                distribution = {}
                for row in rows:
                    for k in row["active_k_by_class"].values():
                        distribution[str(k)] = distribution.get(str(k), 0) + 1
                groups.append({"dataset": dataset, "variant": variant, "session": session,
                               "metrics": {key: mean_std([r[key] for r in rows]) for key in metrics},
                               "active_k_distribution": distribution})
            if not variant.startswith("adaptive"):
                continue
            for fixed in ("fixed_k2", "fixed_k3", "fixed_k5"):
                differences = {k: [] for k in ("oa", "current_oa", "apd_base_forgetting", "active_k_mean")}
                for seed in study.SEEDS:
                    a = next(r["sessions"][-1] for r in audited if
                             (r["dataset"], r["variant"], r["seed"]) == (dataset, variant, seed))
                    b = next(r["sessions"][-1] for r in audited if
                             (r["dataset"], r["variant"], r["seed"]) == (dataset, fixed, seed))
                    for key in differences:
                        differences[key].append(a[key] - b[key])
                paired.append({"dataset": dataset, "adaptive": variant, "reference": fixed,
                               "session": 3, "adaptive_minus_fixed": {
                                   key: mean_std(values) for key, values in differences.items()}})
    return {"smoke": False, "audited_runs": 54, "groups": groups, "paired_final_differences": paired,
            "units": "accuracy/forgetting are fractions, not percentage points; K is active prototypes/class",
            "caution": "Three seeds, descriptive sample SD, no significance or noninferiority claim. "
                       "Do not choose eta using test OA. Active budget is not allocated memory. "
                       "This grid is not an exactly budget-matched allocation experiment."}


def plot(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(9, 4.5))
    for row, dataset in enumerate(study.DATASETS):
        for column, (metric, label) in enumerate((("oa", "Final seen-class OA (%)"),
                                                  ("current_oa", "Final current-class OA (%)"),
                                                  ("apd_base_forgetting", "Base forgetting (%)"))):
            ax = axes[row, column]
            for family, variants, color, marker in (
                    ("Fixed K", ["fixed_k2", "fixed_k3", "fixed_k5"], "black", "x"),
                    ("Adaptive / LSE", [f"adaptive_eta{t}_lse" for t in ("003", "010", "020")], "#2368a0", "o"),
                    ("Adaptive / LME", [f"adaptive_eta{t}_lme" for t in ("003", "010", "020")], "#b65019", "s")):
                for idx, variant in enumerate(variants):
                    group = next(g for g in report["groups"] if
                                 (g["dataset"], g["variant"], g["session"]) == (dataset, variant, 3))
                    x, y = group["metrics"]["active_k_mean"], group["metrics"][metric]
                    ax.errorbar(x["mean"], 100 * y["mean"], xerr=x["std"], yerr=100 * y["std"],
                                color=color, marker=marker, capsize=2, linestyle="none",
                                label=family if idx == 0 else None, markersize=4)
                    label_value = variant[-1] if variant.startswith("fixed") else str(study.VARIANTS[variant][2])
                    ax.annotate(label_value, (x["mean"], 100 * y["mean"]),
                                xytext=(3, 4), textcoords="offset points", fontsize=6, color=color)
            ax.set(xlabel="Active prototypes per class", ylabel=label, title=dataset)
            ax.set_xlim(.8, 5.2)
            ax.grid(alpha=.2)
            ax.spines[["top", "right"]].set_visible(False)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .93))
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output / f"capacity_tradeoff.{ext}", dpi=250, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, default=study.ROOT / "capacity_outputs")
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()
    report = summarize(args.output_root)
    study.save_json(args.output_root / "capacity_report.json", report)
    for group in report["groups"]:
        if group["session"] != 3:
            continue
        metrics = group["metrics"]
        oa, budget = metrics["oa"], metrics["active_k_mean"]
        print(f"{group['dataset']:12} {group['variant']:24} "
              f"OA={100*oa['mean']:.2f}+/-{100*oa['std']:.2f}%  "
              f"active K={budget['mean']:.3f}+/-{budget['std']:.3f}")
    if args.plot:
        plot(report, args.output_root)


if __name__ == "__main__":
    main()
