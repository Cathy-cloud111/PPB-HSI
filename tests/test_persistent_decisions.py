"""Synthetic bookkeeping tests; these are not scientific result evidence."""
import copy
import unittest

import numpy as np
import torch

import diagnose_persistent_decisions as decisions
import run_capacity_experiments as provenance


class DecisionTests(unittest.TestCase):
    def test_stage_partition(self):
        groups = decisions.stage_groups([[1, 2], [3], [4]], 4)
        self.assertEqual(groups["middle_increment"], [3])
        for bad in ([[1], [1], [3]], [[1], [2]], [[1], [2], [4]]):
            with self.assertRaises(ValueError):
                decisions.stage_groups(bad, 3)

    def test_corrected_damaged_and_wrong_to_wrong(self):
        y = np.array([0, 0, 0, 1, 1])
        a = np.array([1, 0, 1, 1, 0])
        b = np.array([0, 1, 2, 1, 0])
        delta = decisions.changes(y, a, b, [1, 2])
        self.assertEqual([delta[k] for k in ("corrected", "damaged", "both_correct",
                         "both_wrong_changed", "both_wrong_unchanged")], [1, 1, 1, 1, 1])
        self.assertEqual(delta["prediction_changed"], 3)
        self.assertEqual(delta["net_oa_pp"], 0)

    def test_group_uses_ground_truth_not_prediction(self):
        y, a, b = map(np.array, ([0, 1, 2], [2, 0, 2], [0, 1, 0]))
        d = decisions.changes(y, a, b, [1])
        self.assertEqual(d["support"], 1)
        self.assertEqual(d["corrected"], 1)
        self.assertEqual(d["net_oa_pp"], 100)

    def test_confusion_strict_and_invalid(self):
        y = np.array([0, 1, 1])
        cm = decisions.confusion(y, np.array([0, 0, 1]), 2)
        np.testing.assert_array_equal(cm, [[1, 0], [1, 1]])
        decisions.require_exact_confusion(cm, cm.copy())
        with self.assertRaises(ValueError):
            decisions.require_exact_confusion(cm, np.zeros((2, 2)))
        for p in (np.array([0, 1]), np.array([0, 1, 2]), np.array([0., 1., 1.])):
            with self.assertRaises(ValueError):
                decisions.confusion(y, p, 2)

    def test_summary_net_agrees_with_accuracy_and_partition(self):
        y = np.array([0, 0, 1, 2, 2])
        preds = {"head": np.array([1, 0, 2, 0, 2]),
                 "raw_fusion": np.array([0, 0, 1, 0, 1]),
                 "ipoc_fusion": np.array([0, 1, 1, 2, 1])}
        groups = decisions.stage_groups([[1], [2], [3]], 3)
        report = decisions.summarize_predictions(y, preds, groups, 3)
        for before, after in decisions.CONTRASTS:
            delta = report["contrasts"][f"{before}_to_{after}"]
            for group, stats in delta["groups"].items():
                net = 100 * (report["scores"][after][group]["oa"] - report["scores"][before][group]["oa"])
                self.assertAlmostEqual(net, stats["net_oa_pp"])
                self.assertEqual(stats["support"], sum(stats[k] for k in (
                    "corrected", "damaged", "both_correct", "both_wrong_changed", "both_wrong_unchanged")))
            self.assertEqual(delta["groups"]["all_seen"]["support"],
                             sum(delta["groups"][g]["support"] for g in (
                                 "base", "middle_increment", "last_increment")))

    def fixture(self):
        jobs = provenance.jobs_for(provenance.DATASETS, (42,), ("persistent_er", "persistent_full"))
        plan = {"smoke": True, "requests": [{"job": j} for j in jobs]}
        queue = {"smoke": True, "state": "complete", "completed": [j["name"] for j in jobs]}
        result = {"smoke": True, "complete": True, "runs": jobs}
        pairs = {"smoke": True, "pairs": [{"dataset": d, "seed": 42, "passed": True}
                                           for d in provenance.DATASETS]}
        return plan, queue, result, pairs

    def test_source_smoke_formal_and_missing_run_refused(self):
        f = self.fixture()
        self.assertEqual(len(decisions.validate_source(*f, True)), 2)
        with self.assertRaises(ValueError):
            decisions.validate_source(*f, False)
        broken = copy.deepcopy(f)
        broken[1]["completed"].pop()
        with self.assertRaises(ValueError):
            decisions.validate_source(*broken, True)
        broken = copy.deepcopy(f)
        broken[3]["pairs"][0]["passed"] = False
        with self.assertRaises(ValueError):
            decisions.validate_source(*broken, True)

    def test_state_hash_includes_buffers(self):
        model = torch.nn.BatchNorm1d(2)
        h = decisions.state_hash(model)
        self.assertEqual(h, decisions.state_hash(model))
        model.running_mean[0] = 1
        self.assertNotEqual(h, decisions.state_hash(model))

    def test_aggregate_smoke_sd_not_estimated(self):
        groups = decisions.stage_groups([[1], [2], [3]], 3)
        y = np.array([0, 1, 2])
        stats = decisions.summarize_predictions(y, {m: y for m in decisions.MODES}, groups, 3)
        run = {"job": {"dataset": "synthetic"}, "groups": groups, **stats}
        d = decisions.aggregate([run])["synthetic"]["head_to_raw_fusion"]["all_seen"]
        self.assertIsNone(d["net_oa_pp"]["sample_sd"])


if __name__ == "__main__":
    unittest.main()
