"""Algorithm and queue invariants; synthetic values are never paper evidence."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

import main_icarl_hsi as icarl
import run_icarl_hsi as runner
import run_capacity_experiments as provenance
import summarize_icarl_hsi as reporter


class ICaRLTests(unittest.TestCase):
    def arguments(self, smoke=False, dataset="Houston2013"):
        job = {"name": "test", "dataset": dataset, "seed": 42}
        return icarl.get_args_parser().parse_args(runner.build_command(
            job, Path("unused"), "cpu", smoke=smoke)[3:])

    def test_unique_six_job_plan(self):
        jobs = provenance.jobs_for(provenance.DATASETS, provenance.SEEDS, ("icarl_hsi",))
        self.assertEqual(len(jobs), 6)
        self.assertEqual(len({j["name"] for j in jobs}), 6)

    def test_formal_declared_parameters(self):
        for dataset, total in (("Houston2013", 1500), ("PaviaU", 900)):
            args = self.arguments(dataset=dataset)
            self.assertEqual(args.memory_budget, total)
            self.assertEqual(args.epochs, 80)
            self.assertEqual(args.batch_size, 128)
            self.assertEqual(args.hsi_patch_size, 15)
            self.assertEqual(args.lr, .001)
            self.assertFalse(args.smoke)
            self.assertEqual(args.max_train_samples_per_class, 0)

    def test_smoke_is_explicit_and_capped(self):
        args = self.arguments(True)
        self.assertEqual(len(icarl.validate_args(args)), 3)
        self.assertEqual(args.epochs, 1)
        self.assertEqual(args.memory_budget, 60)
        args.epochs = 80
        with self.assertRaises(ValueError):
            icarl.validate_args(args)

    def test_formal_cpu_refused(self):
        with self.assertRaises(RuntimeError):
            icarl.validate_args(self.arguments())

    def test_no_cuda_silent_fallback(self):
        args = self.arguments(True)
        args.device = "cuda"
        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                icarl.validate_args(args)

    def test_old_sigmoid_targets_replace_indicators_on_all_samples(self):
        labels = torch.tensor([0, 2])  # one old and one new example
        probabilities = torch.tensor([[.2, .7], [.8, .6]], requires_grad=True)
        target = icarl.icarl_targets(labels, [1, 2, 3], [1, 2], probabilities)
        torch.testing.assert_close(target, torch.tensor([[.2, .7, 0.], [.8, .6, 1.]]))
        self.assertFalse(target.requires_grad)
        with self.assertRaises(ValueError):
            icarl.icarl_targets(labels, [1, 2, 3], [1, 2])

    def test_noncontiguous_seen_mapping_and_future_gradient(self):
        logits = torch.zeros(2, 5, requires_grad=True)
        target = icarl.icarl_targets(torch.tensor([0, 3]), [1, 4], [])
        F.binary_cross_entropy_with_logits(logits[:, [0, 3]], target).backward()
        self.assertTrue((logits.grad[:, [1, 2, 4]] == 0).all())
        torch.testing.assert_close(target, torch.eye(2))

    def test_herding_is_ordered_unique_capped_and_deterministic(self):
        features = torch.tensor([[1., 0.], [1., 1.], [0., 1.]])
        selected = icarl.herding_indices(features, 10)
        self.assertEqual(selected[0], 1)  # exact normalized class direction
        self.assertEqual(len(set(selected)), 3)
        self.assertEqual(selected, icarl.herding_indices(features, 10))
        self.assertEqual(icarl.herding_indices(features, 1), selected[:1])

    def test_herding_matches_bruteforce_spherical_objective(self):
        features = F.normalize(torch.tensor([[2., 1.], [1., 4.], [3., 2.], [4., .2]]).double(), dim=1)
        target = F.normalize(features.mean(0), dim=0)
        available, expected, total = list(range(4)), [], torch.zeros(2, dtype=torch.double)
        for _ in range(3):
            chosen = min(available, key=lambda i: float((F.normalize(total + features[i], dim=0) - target).square().sum()))
            expected.append(chosen)
            available.remove(chosen)
            total += features[chosen]
        self.assertEqual(icarl.herding_indices(features, 3), expected)

    def test_prefix_reduction_and_image_only_replay(self):
        images = torch.arange(12.).reshape(3, 1, 2, 2)
        original = {4: {"images": images, "coordinates": [[1, 1], [2, 2], [3, 3]]}}
        reduced = icarl.reduce_memory(original, 2)
        self.assertEqual(reduced[4]["coordinates"], [[1, 1], [2, 2]])
        self.assertEqual(len(original[4]["images"]), 3)
        image, label = icarl.MemoryDataset(reduced)[1]
        torch.testing.assert_close(image, images[1])
        self.assertEqual(int(label), 3)
        reduced[4]["images"].zero_()
        self.assertFalse((original[4]["images"] == 0).all())

    def test_nme_uses_means_and_global_class_ids(self):
        features = torch.tensor([[4., .1], [.1, 3.]])
        means = torch.eye(2)
        self.assertEqual(icarl.nme_predict(features, means, [2, 5]).tolist(), [1, 4])
        with self.assertRaises(ValueError):
            icarl.nme_predict(torch.zeros(1, 2), means, [2, 5])

    def test_diagnostic_iterator_does_not_change_train_shuffle(self):
        args = self.arguments(True)
        dataset = torch.utils.data.TensorDataset(torch.arange(25), torch.arange(25))
        first = [x.tolist() for x, _ in icarl.loader(dataset, args, True, 2)]
        list(icarl.loader(dataset, args))
        second = [x.tolist() for x, _ in icarl.loader(dataset, args, True, 2)]
        self.assertEqual(first, second)

    def test_teacher_cache_preserves_index_order(self):
        class FakeModel(torch.nn.Module):
            def forward(self, x):
                return torch.cat((x, -x, 3 * x), dim=1)
        args = self.arguments(True)
        dataset = torch.utils.data.TensorDataset(torch.tensor([[0.], [1.], [2.]]), torch.zeros(3))
        cached = icarl.cache_old_targets(FakeModel(), dataset, [3, 1], args)
        expected = torch.tensor([[0., 0.], [3., 1.], [6., 2.]]).sigmoid()
        torch.testing.assert_close(cached, expected)

    def test_summary_rejects_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "study_results.json").write_text(json.dumps({"smoke": True}))
            (root / "study_plan.json").write_text(json.dumps({"smoke": True}))
            with self.assertRaises(ValueError):
                reporter.summarize(root)

    def test_formal_summary_aggregates_all_seeds_and_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = provenance.jobs_for(provenance.DATASETS, provenance.SEEDS, ("icarl_hsi",))
            requests, runs = [], []
            for job in jobs:
                args = vars(icarl.get_args_parser().parse_args(runner.build_command(job, root, "cuda")[3:]))
                offset = provenance.SEEDS.index(job["seed"]) * .01
                sessions = [{"session": s, **{k: .5 + .1 * s + offset for k in provenance.METRICS}}
                            for s in (1, 2, 3)]
                requests.append({"job": job, "effective_args": args, "identity": {"data_sha256": {}}})
                runs.append({**job, "sessions": sessions})
            (root / "study_plan.json").write_text(json.dumps({"smoke": False, "total_jobs": 6, "requests": requests}))
            (root / "study_results.json").write_text(json.dumps({"smoke": False, "complete": True,
                "planned_jobs": 6, "runs": runs}))
            def checked(folder, request):
                return next(r["sessions"] for r in runs if r["name"] == folder.name)
            with patch.object(runner, "require_resume", side_effect=checked) as verify:
                result = reporter.summarize(root)
            self.assertEqual(verify.call_count, 6)
            self.assertEqual(result["audited_runs"], 6)
            for group in result["groups"]:
                self.assertAlmostEqual(group["metrics"]["oa"]["mean"], .81)
                self.assertAlmostEqual(group["metrics"]["oa"]["sample_sd"], .01)
                self.assertAlmostEqual(group["metrics"]["average_incremental_oa"]["mean"], .71)

    def test_changed_fingerprint_refuses_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps({"fingerprint": "old"}))
            with self.assertRaises(ValueError):
                runner.require_resume(root, {"fingerprint": "new"})


if __name__ == "__main__":
    unittest.main()
