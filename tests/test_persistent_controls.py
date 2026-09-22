"""Synthetic invariants for matched persistent image-memory controls."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import Dataset

import main_houston_cls as engine
import main_persistent_controls as entry
import persistent_control_audit as audit
import run_capacity_experiments as provenance
import run_persistent_controls as runner
import summarize_persistent_controls as summary


class TinyCurrent(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        y, x, c = self.samples[i]
        return torch.full((1, 2, 2), float(10*y+x)), torch.tensor(c-1)


class PersistentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def args(self, variant="persistent_er", smoke=True, dataset="Houston2013"):
        job = {"name": "test", "dataset": dataset, "seed": 42, "variant": variant}
        return entry.get_args_parser().parse_args(runner.build_command(job, Path("unused"), "cpu", smoke=smoke)[3:])

    def pair_fixture(self):
        requests = []
        for variant in runner.VARIANTS:
            args = vars(self.args(variant))
            identity = {"protocol": audit.scientific_args(args), "code_sha256": {},
                        "data_sha256": {}, "runtime": {}, "smoke": True}
            requests.append({"job": {"dataset": "Houston2013", "seed": 42},
                "effective_args": args, "identity": identity, "fingerprint": provenance.fingerprint(identity)})
        rows = [{"session": idx, **{field: "same" for field in audit.PAIR_FIELDS}} for idx in (1, 2, 3)]
        return requests, rows

    def test_twelve_unique_formal_jobs(self):
        jobs = provenance.jobs_for(provenance.DATASETS, provenance.SEEDS, runner.VARIANTS)
        self.assertEqual(len(jobs), 12)
        self.assertEqual(len({job["name"] for job in jobs}), 12)

    def test_only_declared_components_differ(self):
        er, full = [audit.scientific_args(vars(self.args(v, False))) for v in runner.VARIANTS]
        self.assertEqual({k for k in er if er[k] != full[k]},
            {"persistent_variant", "use_prototypes", "adaptive_prototypes", "use_prototype_offsets", "lambda_old_logit_distill"})
        for args in (er, full):
            self.assertEqual(args["memory_budget"], 1500)
            self.assertEqual(args["epochs"], 80)
            self.assertEqual(args["batch_size"], 128)
            self.assertEqual(args["freeze_backbone_after_base"], 1)
            self.assertEqual(args["freeze_shared_prompts_after_base"], 1)
            self.assertEqual(args["class_weight"], "balanced")
        self.assertEqual(self.args(smoke=False, dataset="PaviaU").memory_budget, 900)

    def test_full_settings_preserve_predeclared_method(self):
        actual = audit.scientific_args(vars(self.args("persistent_full", False)))
        job = {"name": "test", "dataset": "Houston2013", "seed": 42, "variant": "adaptive_eta003_lse"}
        reference = provenance.scientific_args(vars(engine.get_args_parser().parse_args(
            provenance.build_command(job, Path("unused"), "cpu")[3:])))
        for key, value in reference.items():
            self.assertEqual(actual[key], value, key)

    def test_smoke_cpu_and_formal_cpu_guard(self):
        entry.validate_args(self.args())
        with self.assertRaises(RuntimeError):
            entry.validate_args(self.args(smoke=False))
        bad = self.args()
        bad.epochs = 80
        with self.assertRaises(ValueError):
            entry.validate_args(bad)

    def test_priority_is_model_independent_and_input_order_independent(self):
        samples = [(i, 1, 1) for i in range(20)]
        shuffled = samples[::-1]
        a = [samples[i] for i in entry.priority_order(samples, 1, 42)]
        b = [shuffled[i] for i in entry.priority_order(shuffled, 1, 42)]
        self.assertEqual(a, b)
        self.assertEqual(len(set(a)), 20)
        self.assertNotEqual(a, [samples[i] for i in entry.priority_order(samples, 1, 123)])

    def test_memory_is_prefix_reduced_without_old_reselection(self):
        first = TinyCurrent([(i, 0, 1) for i in range(6)])
        memory = entry.update_memory({}, first, [1], 4, 42)
        saved_images, saved_coords = memory[1]["images"].clone(), copy.deepcopy(memory[1]["coordinates"])
        second = TinyCurrent([(i, 2, 2) for i in range(6)])
        updated = entry.update_memory(memory, second, [2], 2, 42)
        self.assertEqual(updated[1]["coordinates"], saved_coords[:2])
        torch.testing.assert_close(updated[1]["images"], saved_images[:2])
        self.assertEqual(len(memory[1]["images"]), 4)
        with self.assertRaises(ValueError):
            entry.update_memory(memory, first, [1], 2, 42)

    def test_replay_reads_images_not_old_scene_data(self):
        memory = {1: {"images": torch.full((2, 1, 2, 2), 777.), "coordinates": [[1, 1], [2, 2]]}}
        current = TinyCurrent([(4, 4, 2)])
        dataset = entry.PersistentTrainingDataset(current, memory, [1, 2])
        image, label = dataset[1]
        self.assertEqual(float(image.mean()), 777.)
        self.assertEqual(int(label), 0)
        self.assertEqual(dataset.samples, [(4, 4, 2), (1, 1, 1), (2, 2, 1)])
        self.assertEqual(dataset.class_counts(), {1: 2, 2: 1})

    def test_same_sampler_ignores_global_rng_and_extra_iterators(self):
        a = entry.EpochOrderSampler(20, 42, 2)
        b = entry.EpochOrderSampler(20, 42, 2)
        for epoch in (1, 2):
            ha = a.set_epoch(epoch)
            torch.rand(77)
            list(entry.make_loader(TinyCurrent([(i, 0, 1) for i in range(20)]), self.args(), 77, shuffle=True))
            hb = b.set_epoch(epoch)
            self.assertEqual(ha, hb)
            self.assertEqual(list(a), list(b))
        self.assertEqual(sorted(list(a)), list(range(20)))

    def test_common_model_initialization_matches(self):
        hashes = []
        for variant in runner.VARIANTS:
            engine.seed_everything(42)
            model = entry.build_model(self.args(variant), 2)
            hashes.append(entry.common_model_hash(model))
        self.assertEqual(hashes[0], hashes[1])

    def test_pair_rejects_changed_memory_or_epoch_order(self):
        requests, rows = self.pair_fixture()
        self.assertTrue(audit.check_pair(*requests, rows, copy.deepcopy(rows))["passed"])
        for field in ("memory_images_sha256", "train_samples_sha256", "epoch_order_sha256s", "initial_common_model_sha256"):
            altered = copy.deepcopy(rows)
            altered[1][field] = "different"
            with self.assertRaises(ValueError):
                audit.check_pair(*requests, rows, altered)

    def test_pair_rejects_undeclared_scientific_change(self):
        requests, rows = self.pair_fixture()
        requests[1]["effective_args"]["lr"] = .05
        requests[1]["identity"]["protocol"] = audit.scientific_args(requests[1]["effective_args"])
        requests[1]["fingerprint"] = provenance.fingerprint(requests[1]["identity"])
        with self.assertRaises(ValueError):
            audit.check_pair(*requests, rows, rows)

    def test_audit_current_selection_matches_original_dataset_selection(self):
        labels = np.array([[1, 1, 2], [2, 3, 3], [1, 2, 3]])
        hsi = np.ones((3, 3, 2), dtype=np.float32)
        from unittest.mock import patch
        with patch("datasets.houston_cls.load_mat_array", return_value=labels):
            dataset = entry.HoustonPatchClassification("unused", "unused", [1, 3],
                hsi_array=hsi, max_samples_per_class=2, seed=42)
        self.assertEqual(dataset.samples, audit.selected_current_samples(labels, [1, 3], 2, 42))

    def test_summary_refuses_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename in ("study_results.json", "study_plan.json", "queue_status.json"):
                (root / filename).write_text(json.dumps({"smoke": True}))
            with self.assertRaises(ValueError):
                summary.summarize(root)

    def test_complete_summary_preserves_seed_pairs_and_units(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = provenance.jobs_for(provenance.DATASETS, provenance.SEEDS, runner.VARIANTS)
            requests, runs = [], []
            for job in jobs:
                args = vars(self.args(job['variant'], False, job['dataset']))
                args['seed'] = job['seed']
                identity = {'smoke': False, 'protocol': audit.scientific_args(args),
                            'code_sha256': {}, 'data_sha256': {}, 'runtime': {}}
                requests.append({'job': job, 'effective_args': args, 'identity': identity,
                                 'fingerprint': provenance.fingerprint(identity)})
                offset = .01 if job['variant'] == 'persistent_full' else 0.
                seed_offset = .02 * list(provenance.SEEDS).index(job['seed'])
                rows = [{'session': i, **{field: 'same' for field in audit.PAIR_FIELDS},
                         **{metric: .5 + seed_offset + offset for metric in provenance.METRICS}}
                        for i in (1, 2, 3)]
                runs.append({**job, 'sessions': rows})
            pairs = []
            for dataset in provenance.DATASETS:
                for seed in provenance.SEEDS:
                    pair_requests = [next(r for r in requests if r['job']['dataset'] == dataset
                        and r['job']['seed'] == seed and r['job']['variant'] == variant)
                        for variant in runner.VARIANTS]
                    pair_rows = [next(r['sessions'] for r in runs if r['name'] == req['job']['name'])
                                 for req in pair_requests]
                    pairs.append(audit.check_pair(*pair_requests, *pair_rows))
            payloads = {
                'study_plan.json': {'smoke': False, 'total_jobs': 12, 'requests': requests},
                'study_results.json': {'smoke': False, 'complete': True, 'planned_jobs': 12, 'runs': runs},
                'queue_status.json': {'smoke': False, 'state': 'complete',
                                      'completed': [j['name'] for j in jobs]},
                'pair_checks.json': {'smoke': False, 'pairs': pairs}}
            for filename, value in payloads.items():
                (root / filename).write_text(json.dumps(value))
            def checked(folder, request):
                return next(r['sessions'] for r in runs if r['name'] == folder.name)
            with patch.object(runner, 'require_resume', side_effect=checked):
                report = summary.summarize(root)
            self.assertEqual(report['audited_runs'], 12)
            self.assertEqual(len(report['groups']), 4)
            self.assertEqual(len(report['pair_checks']), 6)
            for contrast in report['paired_differences']:
                self.assertAlmostEqual(contrast['metrics']['oa']['mean'], 1.)
                self.assertEqual(contrast['metrics']['oa']['seeds'], list(provenance.SEEDS))
                self.assertEqual(contrast['metrics']['oa']['n'], 3)
            for group in report['groups']:
                self.assertAlmostEqual(group['metrics']['oa']['sample_sd'], .02)


if __name__ == "__main__":
    unittest.main()
