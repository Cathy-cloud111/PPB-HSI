"""Synthetic unit checks, never paper evidence."""
import copy
from pathlib import Path
import sys
import json
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import main_houston_cls as engine
import main_protocol_controls as entry
import run_protocol_controls as runner
import run_capacity_experiments as study
import summarize_protocol_controls as reporter


class Tiny(torch.nn.Module):
    def __init__(self, logits=(.3, -.2, .9, .8, .1)):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor(logits, dtype=torch.float64))
        self.contexts = []

    def forward_for_session(self, images, active_class_ids, return_features=False):
        self.contexts.append(list(active_class_ids))
        logits = self.logits.expand(len(images), -1)
        return logits, logits.new_tensor(0.)


class Controls(unittest.TestCase):
    def epoch(self, scope, kd=False, margin=0., original=False):
        model = Tiny()
        teacher = Tiny((-.2, .4, .2, .1, .3)) if kd else None
        loader = [(torch.ones(2, 1), torch.tensor([2, 3]))]
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        kwargs = dict(model=model, loader=loader, optimizer=optimizer, device="cpu", epoch=1,
                      class_weights=torch.ones(5, dtype=torch.float64), train_class_ids=[3, 4],
                      prompt_loss_weight=.1, confusion_margin_weight=margin,
                      confusion_margin_target_class_ids=[3, 4], teacher_model=teacher,
                      distill_class_ids=[1, 2], old_logit_distill_weight=1. if kd else 0.,
                      old_logit_distill_scope="all")
        if original:
            output = engine.train_one_epoch(**kwargs)
        else:
            output = entry.train_one_epoch(**kwargs, supervised_class_scope=scope)
        return model, output

    def test_default_grid(self):
        jobs = study.jobs_for(study.DATASETS, study.SEEDS, runner.VARIANTS)
        self.assertEqual(len(jobs), 30)
        self.assertEqual(len({j['name'] for j in jobs}), 30)

    def test_scope_union_and_future_exclusion(self):
        self.assertEqual(entry.supervised_classes([3, 4], [1, 2], "seen"), [1, 2, 3, 4])
        self.assertEqual(entry.supervised_classes([3, 4], [1, 2], "sampled"), [3, 4])
        with self.assertRaises(ValueError):
            entry.supervised_classes([3], [1], "invalid")

    def test_ce_gradient_only(self):
        sampled, _ = self.epoch("sampled")
        seen, _ = self.epoch("seen")
        self.assertTrue(torch.equal(sampled.logits.grad[:2], torch.zeros(2, dtype=torch.float64)))
        self.assertTrue(torch.all(seen.logits.grad[:2] > 0))
        self.assertEqual(float(seen.logits.grad[4]), 0.)
        self.assertEqual(sampled.contexts, [[3, 4]])
        self.assertEqual(seen.contexts, [[3, 4]])

    def test_original_epoch_equivalence(self):
        for kd in (False, True):
            a, first = self.epoch("sampled", kd=kd, margin=.05, original=True)
            b, second = self.epoch("sampled", kd=kd, margin=.05)
            self.assertEqual(first, second)
            self.assertTrue(torch.equal(a.logits, b.logits))
            self.assertTrue(torch.equal(a.logits.grad, b.logits.grad))

    def test_kd_term_not_changed_by_ce_scope(self):
        gradients = {}
        for scope in ("sampled", "seen"):
            with_kd, _ = self.epoch(scope, kd=True)
            without_kd, _ = self.epoch(scope, kd=False)
            gradients[scope] = with_kd.logits.grad-without_kd.logits.grad
            self.assertEqual(with_kd.contexts, [[3, 4]])
            self.assertEqual(float(with_kd.logits.grad[4]), 0.)
        self.assertTrue(torch.allclose(gradients['sampled'], gradients['seen'], atol=1e-15))

    def test_margin_mask_not_changed(self):
        gradients = {}
        for scope in ("sampled", "seen"):
            with_margin, _ = self.epoch(scope, margin=.05)
            without_margin, _ = self.epoch(scope, margin=0.)
            gradients[scope] = with_margin.logits.grad-without_margin.logits.grad
        self.assertTrue(torch.allclose(gradients['sampled'], gradients['seen'], atol=1e-15))

    def parsed(self, variant, smoke=False):
        job = {"dataset":"Houston2013", "variant":variant, "seed":42, "name":variant}
        cmd = runner.build_command(job, Path("unused"), "cpu", smoke=smoke)
        return vars(entry.get_args_parser().parse_args(cmd[3:]))

    def test_mask_pairs_change_scope_only(self):
        for name in ("ft", "lwf"):
            a, b = self.parsed(name+'_sampled'), self.parsed(name+'_seen')
            differences = {key for key in a if a[key] != b[key]}
            self.assertEqual(differences, {"output_dir", "supervised_class_scope"})
            self.assertEqual(a['incremental_train'], 'current')
            self.assertEqual(a['replay_old_samples_per_class'], 0)
            self.assertEqual(a['epochs'], 80)

    def test_er_matches_capacity_common_settings(self):
        job = {"dataset":"Houston2013", "variant":"fixed_k5", "seed":42, "name":"reference"}
        ref = vars(engine.get_args_parser().parse_args(study.build_command(job, Path('unused'), 'cpu')[3:]))
        er = self.parsed('er_replay')
        del er['supervised_class_scope']
        changed = {key for key in er if er[key] != ref[key]}
        self.assertEqual(changed, {'output_dir', 'use_prototypes', 'use_prototype_offsets', 'lambda_old_logit_distill'})
        self.assertEqual(er['incremental_train'], 'replay')
        self.assertEqual(er['replay_old_samples_per_class'], 100)

    def test_smoke_is_capped_and_distinct(self):
        args = self.parsed('ft_seen', smoke=True)
        self.assertEqual(args['epochs'], 1)
        self.assertEqual(args['max_train_samples_per_class'], 8)
        self.assertEqual(args['max_test_samples_per_class'], 12)

    def test_import_does_not_change_original_engine(self):
        self.assertIs(engine.get_args_parser, entry.ORIGINAL_PARSER)
        self.assertNotIn('supervised_class_scope', vars(engine.get_args_parser().parse_args([])))

    def test_entry_passes_parsed_args_to_engine(self):
        original_epoch = engine.train_one_epoch
        try:
            with patch.object(sys, 'argv', ['entry', '--supervised_class_scope', 'seen']), patch.object(engine, 'main') as launch:
                entry.main()
                launch.assert_called_once()
                self.assertEqual(launch.call_args.args[0].supervised_class_scope, 'seen')
        finally:
            engine.train_one_epoch = original_epoch

    def test_summary_rejects_smoke_and_partial(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            for smoke, complete in ((True, True), (False, False)):
                (output/'study_results.json').write_text(json.dumps({
                    'smoke':smoke, 'complete':complete, 'planned_jobs':30, 'runs':[]}))
                with self.assertRaises(ValueError):
                    reporter.summarize(output)


if __name__ == "__main__":
    unittest.main()
