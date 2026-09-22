"""Unit tests; synthetic data here are never experimental paper evidence."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import main_houston_cls as engine
import run_capacity_experiments as study
import summarize_capacity_experiments as reporter
from datasets.houston_cls import confusion_to_metrics, parse_class_sessions


class CapacityTests(unittest.TestCase):
    def test_predeclared_grid(self):
        jobs = study.jobs_for(study.DATASETS, study.SEEDS, study.VARIANTS)
        self.assertEqual(len(jobs), 54)
        self.assertEqual(len({j["name"] for j in jobs}), 54)

    def test_only_capacity_controls_change(self):
        values = []
        for variant in study.VARIANTS:
            job = {"name": variant, "dataset": "Houston2013", "variant": variant, "seed": 42}
            cmd = study.build_command(job, Path("unused"), "cpu")
            args = vars(engine.get_args_parser().parse_args(cmd[3:]))
            self.assertEqual(args["epochs"], 80)
            self.assertEqual(args["prototype_offset_epochs"], 10)
            self.assertEqual(args["replay_old_samples_per_class"], 100)
            self.assertEqual(args["lambda_old_logit_distill"], .05)
            self.assertEqual(args["max_train_samples_per_class"], 0)
            controls = {"adaptive_prototypes", "num_prototypes_per_class",
                        "adaptive_elbow_min_gain", "prototype_pooling"}
            values.append({k: v for k, v in study.scientific_args(args).items() if k not in controls})
        self.assertTrue(all(v == values[0] for v in values))

    def test_smoke_protocol_not_formal(self):
        job = study.jobs_for(["PaviaU"], [42], ["fixed_k2"])[0]
        formal = vars(engine.get_args_parser().parse_args(study.build_command(job, ROOT, "cpu")[3:]))
        smoke = vars(engine.get_args_parser().parse_args(study.build_command(job, ROOT, "cpu", smoke=True)[3:]))
        self.assertEqual(smoke["epochs"], 1)
        self.assertEqual(smoke["max_test_samples_per_class"], 12)
        self.assertNotEqual(study.fingerprint(study.scientific_args(formal)),
                            study.fingerprint(study.scientific_args(smoke)))

    def test_elbow_stopping_and_saturation(self):
        features = torch.ones(10, 4)
        with patch.object(engine, "compute_feature_prototypes", return_value=(features[:2], torch.ones(2))):
            for errors, expected in (([1., .99, .98, .97], 2), ([1., .7, .4, .2], 5),
                                     ([1., .8, .85, .7], 3)):
                with patch.object(engine, "prototype_assignment_error", side_effect=errors):
                    actual = engine.choose_adaptive_num_prototypes(
                        features, 5, min_prototypes=2, mode="elbow", elbow_min_gain=.1)
                    self.assertEqual(actual, expected)

    def test_pooling_uniform_k_inference_invariant(self):
        torch.manual_seed(8)
        model = SimpleNamespace(prototypes=torch.randn(3, 3, 8), prototype_counts=torch.ones(3, 3))
        logits, features = torch.randn(12, 3), torch.randn(12, 8)
        lse = engine.add_prototype_logits(model, logits, features, [1, 2, 3], alpha=.7, pooling="logsumexp")
        lme = engine.add_prototype_logits(model, logits, features, [1, 2, 3], alpha=.7, pooling="logmeanexp")
        torch.testing.assert_close(lse - lme, torch.full_like(lse, .7 * np.log(3)), atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(lse.softmax(1), lme.softmax(1), atol=1e-6, rtol=1e-5)

    def test_pooling_variable_k_count_bias(self):
        model = SimpleNamespace(prototypes=torch.ones(2, 5, 4),
                                prototype_counts=torch.tensor([[1., 0., 0., 0., 0.], [1., 1., 1., 1., 1.]]))
        logits, features = torch.zeros(1, 2), torch.ones(1, 4)
        lse = engine.add_prototype_logits(model, logits, features, [1, 2], pooling="logsumexp")
        lme = engine.add_prototype_logits(model, logits, features, [1, 2], pooling="logmeanexp")
        self.assertGreater(float(lse[0, 1]), float(lse[0, 0]))
        torch.testing.assert_close(lme[:, 0], lme[:, 1])

    def fixture(self, folder):
        job = study.jobs_for(["PaviaU"], [42], ["fixed_k2"])[0]
        args = vars(engine.get_args_parser().parse_args(study.build_command(job, folder, "cpu", smoke=True)[3:]))
        seen, rows = [], []
        for idx, current in enumerate(parse_class_sessions(args["sessions"]), 1):
            seen += current
            cm = np.zeros((9, 9), dtype=np.int64)
            cm[np.array(seen) - 1, np.array(seen) - 1] = 12
            row = {"session": idx, **{k: 1. for k in ("oa", "aa", "kappa", "current_oa", "base_oa")},
                   "apd_base_oa_raw": 0., "apd_base_forgetting": 0., "test_samples": int(cm.sum()),
                   "active_prototype_k_mean": 2.,
                   "active_prototype_k_by_class": json.dumps({str(c): 2 for c in seen})}
            counts = torch.zeros(9, 2)
            counts[np.array(seen) - 1] = 4
            torch.save({"model": {"prototype_counts": counts}, "args": args, "session": idx,
                        "seen_classes": list(seen), "metrics": row}, folder / f"session_{idx}.pth")
            np.save(folder / f"confusion_session_{idx}.npy", cm)
            rows.append(row)
        study.save_json(folder / "metrics.json", rows)
        return args

    def test_audit_and_safe_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            args = self.fixture(folder)
            sessions = study.audit_run(folder, args)
            self.assertEqual(sessions[-1]["active_prototypes"], 18)
            request = {"fingerprint": "original", "effective_args": args}
            study.save_json(folder / "manifest.json", request)
            with self.assertRaisesRegex(ValueError, "unfinished"):
                study.require_resume(folder, request)
            study.save_json(folder / "completed.json", {"fingerprint": "original", "sessions": sessions})
            self.assertEqual(study.require_resume(folder, request), sessions)
            with self.assertRaisesRegex(ValueError, "changed"):
                study.require_resume(folder, {**request, "fingerprint": "different"})
            rows = json.loads((folder / "metrics.json").read_text())
            rows[-1]["oa"] = .99
            study.save_json(folder / "metrics.json", rows)
            with self.assertRaisesRegex(ValueError, "recomputed"):
                study.require_resume(folder, request)

    def test_reject_changed_checkpoint_config(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            args = self.fixture(folder)
            altered = copy.deepcopy(args)
            altered["replay_old_samples_per_class"] = 101
            with self.assertRaisesRegex(ValueError, "different training arguments"):
                study.audit_run(folder, altered)

    def test_engine_runtime_null_filter(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            args = self.fixture(folder)
            for idx in (1, 2, 3):
                file = folder / f"session_{idx}.pth"
                saved = torch.load(file, map_location="cpu", weights_only=True)
                saved["args"]["sample_filter"] = None
                torch.save(saved, file)
            self.assertEqual(len(study.audit_run(folder, args)), 3)
            self.assertNotEqual(study.scientific_args({"sample_filter": {"train": [1]}}), {})

    def test_report_refuses_smoke_and_partial(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            for smoke, complete, planned in ((True, True, 54), (False, False, 54), (False, True, 4)):
                study.save_json(output / "study_results.json", {"smoke": smoke, "complete": complete,
                                "planned_jobs": planned, "runs": []})
                with self.assertRaisesRegex(ValueError, "54 formal"):
                    reporter.summarize(output)


if __name__ == "__main__":
    unittest.main()
