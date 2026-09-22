"""Audit the complete 30-job control grid; no automatic main-table replacement."""
import argparse
import json
from pathlib import Path
import numpy as np
import main_protocol_controls as entry
import run_protocol_controls as controls
import run_capacity_experiments as audit


def statistics(values):
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)), "n":len(values)}


def summarize(output):
    saved = json.loads((output/"study_results.json").read_text())
    jobs = audit.jobs_for(audit.DATASETS, audit.SEEDS, controls.VARIANTS)
    if saved["smoke"] or not saved["complete"] or saved["planned_jobs"] != 30 or len(saved["runs"]) != 30:
        raise ValueError("Require all 30 formal controls, not smoke/partial results")
    if {r['name'] for r in saved['runs']} != {j['name'] for j in jobs}:
        raise ValueError("Unexpected job grid")
    checked, manifests = [], []
    for job in jobs:
        folder = output/job['name']
        manifest = json.loads((folder/"manifest.json").read_text())
        args = manifest['effective_args']
        declared = vars(entry.get_args_parser().parse_args(controls.build_command(
            job, output, args['device'])[3:]))
        if manifest['job'] != job or manifest['identity']['smoke'] or (
            audit.scientific_args(args) != audit.scientific_args(declared)):
            raise ValueError(f"Not a predeclared formal configuration: {job['name']}")
        if manifest['fingerprint'] != audit.fingerprint(manifest['identity']):
            raise ValueError("Invalid identity fingerprint")
        sessions = audit.require_resume(folder, manifest)
        if sessions != next(r for r in saved['runs'] if r['name']==job['name'])['sessions']:
            raise ValueError("Aggregate output differs from checked job")
        checked.append({**job, 'sessions':sessions})
        manifests.append(manifest)
    for dataset in audit.DATASETS:
        identities = [m['identity'] for m in manifests if m['job']['dataset']==dataset]
        for field in ('code_sha256', 'data_sha256', 'runtime'):
            if any(i[field] != identities[0][field] for i in identities):
                raise ValueError(f"Mixed {field} within {dataset}")
        rows = [np.load(output/j['name']/"confusion_session_3.npy").sum(axis=1)
                for j in jobs if j['dataset']==dataset]
        if any(not np.array_equal(row, rows[0]) for row in rows):
            raise ValueError("Different final evaluation class counts")
    groups, paired = [], []
    def row(dataset, variant, seed, session=3):
        return next(r for r in checked if (r['dataset'], r['variant'], r['seed']) ==
                    (dataset, variant, seed))['sessions'][session-1]
    for dataset in audit.DATASETS:
        for variant in controls.VARIANTS:
            for session in (1, 2, 3):
                groups.append({'dataset':dataset, 'variant':variant, 'session':session,
                               'metrics':{k:statistics([row(dataset, variant, seed, session)[k]
                                                       for seed in audit.SEEDS]) for k in audit.METRICS}})
        for family in ('ft', 'lwf'):
            paired.append({'dataset':dataset, 'comparison':family+'_seen minus '+family+'_sampled',
                           'session':3, 'differences':{k:statistics([
                               row(dataset, family+'_seen', seed)[k]-row(dataset, family+'_sampled', seed)[k]
                               for seed in audit.SEEDS]) for k in audit.METRICS}})
    return {'audited_runs':30, 'smoke':False, 'groups':groups, 'paired_ce_mask_differences':paired,
            'units':'Accuracy and forgetting are fractions; multiply by100 for percentage points.',
            'caution':'Three seeds, descriptive sample SD, not strong external-method baselines. '
                       'Do not automatically replace old main-table rows or select the best test configuration.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=controls.ROOT/'protocol_outputs')
    args = parser.parse_args()
    report = summarize(args.output_root)
    audit.save_json(args.output_root/'protocol_report.json', report)
    for g in report['groups']:
        if g['session'] == 3:
            oa = g['metrics']['oa']
            print(f"{g['dataset']:12} {g['variant']:12} OA={100*oa['mean']:.2f}+/-{100*oa['std']:.2f}%")


if __name__ == '__main__':
    main()
