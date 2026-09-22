"""Mechanism checks only; never substitute synthetic checks for paper evidence."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main_houston_cls as engine


class Scoring(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        torch.manual_seed(7)
        self.model = engine.HoustonPDPClassifier(1, num_classes=3,
            num_prototypes_per_class=5, use_prototype_offsets=1,
            prototype_offset_scale=.02, prototype_offset_start_session=2)
        self.model.set_prompt_task_context([1], [2], 2)
        with torch.no_grad():
            self.model.prototypes[:2] = torch.nn.functional.normalize(torch.randn(2,5,256),dim=2)
            self.model.prototype_counts[0,:2] = 1
            self.model.prototype_counts[1,:5] = 1

    def test_lse_cardinality_term_is_exact(self):
        z = torch.randn(4,256)
        head = torch.randn(4,3)
        a = engine.add_prototype_logits(self.model, head,z,[1,2],alpha=.7,pooling='logsumexp')
        b = engine.add_prototype_logits(self.model, head,z,[1,2],alpha=.7,pooling='logmeanexp')
        expected = .7*torch.tensor([2.,5.]).log()
        self.assertTrue(torch.allclose((a-b)[:,:2],expected.expand(4,-1),atol=1e-6))
        self.assertTrue(torch.equal(a[:,2],head[:,2]))

    def test_offset_ce_gradient_is_nonzero_and_inactive_slots_masked(self):
        features = torch.randn(4,256)
        head = torch.randn(4,3)
        fused = engine.add_prototype_logits(self.model,head,features,[1,2],pooling='logsumexp')
        loss = torch.nn.functional.cross_entropy(engine.mask_logits_to_classes(fused,[1,2]),
                                                torch.tensor([0,1,0,1]))
        loss.backward()
        grad = self.model.prototype_offsets.grad
        self.assertGreater(float(grad[:2].abs().sum()),0.)
        self.assertTrue(torch.equal(grad[0,2:],torch.zeros_like(grad[0,2:])))
        self.assertTrue(torch.equal(grad[2],torch.zeros_like(grad[2])))

    def test_calibration_only_updates_offsets_and_restores_flags(self):
        self.model.train()
        before = {k:v.clone() for k,v in self.model.state_dict().items()}
        flags = [p.requires_grad for p in self.model.parameters()]
        batch = [(torch.randn(4,1,15,15),torch.tensor([0,1,0,1]))]
        result = engine.calibrate_prototype_offsets(self.model,batch,'cpu',[1,2],
            torch.ones(3),epochs=2,lr=.001,l2_weight=.01,prototype_alpha=.7)
        self.assertIsNotNone(result)
        self.assertTrue(self.model.training)
        self.assertEqual(flags,[p.requires_grad for p in self.model.parameters()])
        changed = {k for k,v in self.model.state_dict().items() if not torch.equal(v,before[k])}
        self.assertEqual(changed,{'prototype_offsets'})

    def test_actual_evaluate_switch_changes_prediction(self):
        features = torch.zeros(1, 256)
        features[0, 1] = 1.
        head = torch.tensor([[.1, 0., -10.]])
        with torch.no_grad():
            self.model.prototypes.zero_()
            self.model.prototype_counts.zero_()
            self.model.prototype_offsets.zero_()
            self.model.prototypes[0, 0, 0] = 1.
            self.model.prototypes[1, 0, 1] = 1.
            self.model.prototype_counts[:2, 0] = 1.
        batch = [(torch.zeros(1, 1, 15, 15), torch.tensor([1]))]
        with patch.object(engine, 'forward_for_classes', return_value=(head, features, head.new_tensor(0.))):
            baseline = engine.evaluate(self.model, batch, 'cpu', [1, 2], num_classes=3, use_prototypes=False)
            fused = engine.evaluate(self.model, batch, 'cpu', [1, 2], num_classes=3,
                use_prototypes=True, prototype_alpha=.7, prototype_temperature=.2,
                prototype_pooling='logsumexp')
        expected_base, expected_fused = np.zeros((3, 3), dtype=np.int64), np.zeros((3, 3), dtype=np.int64)
        expected_base[1, 0], expected_fused[1, 1] = 1, 1
        np.testing.assert_array_equal(baseline, expected_base)
        np.testing.assert_array_equal(fused, expected_fused)

    def test_radial_offset_is_removed_by_cosine_normalization(self):
        features, head = torch.randn(4, 256), torch.randn(4, 3)
        with torch.no_grad():
            self.model.prototype_offsets.copy_(3.*self.model.prototypes)
        raw = engine.add_prototype_logits(self.model, head, features, [1, 2], calibrated=False)
        calibrated = engine.add_prototype_logits(self.model, head, features, [1, 2], calibrated=True)
        torch.testing.assert_close(raw, calibrated, atol=1e-6, rtol=1e-5)

    def test_scoring_uses_count_positivity_not_cluster_population_weights(self):
        features, head = torch.randn(4, 256), torch.randn(4, 3)
        before = engine.add_prototype_logits(self.model, head, features, [1, 2], pooling='logsumexp')
        with torch.no_grad():
            self.model.prototype_counts.mul_(17.)
        after = engine.add_prototype_logits(self.model, head, features, [1, 2], pooling='logsumexp')
        torch.testing.assert_close(before, after, atol=0, rtol=0)

    def test_base_stage_offsets_are_intentionally_inactive(self):
        self.model.set_prompt_task_context([], [1, 2], 1)
        self.assertFalse(self.model.prototype_offsets_active())
        with patch.object(torch.optim, 'AdamW') as optimizer:
            result = engine.calibrate_prototype_offsets(self.model, [], 'cpu', [1, 2], None)
        self.assertIsNone(result)
        optimizer.assert_not_called()
        self.model.set_prompt_task_context([1], [2], 2)
        self.assertTrue(self.model.prototype_offsets_active())

    def test_elbow_is_relative_gain_and_retains_previous_capacity(self):
        features = torch.ones(10, 4)
        with patch.object(engine, 'compute_feature_prototypes', return_value=(features[:2], torch.ones(2))):
            for eta, expected in ((.25, 5), (.4, 2)):
                with patch.object(engine, 'prototype_assignment_error', side_effect=[1., .75, .5, .25]):
                    actual = engine.choose_adaptive_num_prototypes(features, 5, 2,
                        mode='elbow', elbow_min_gain=eta)
                    self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
