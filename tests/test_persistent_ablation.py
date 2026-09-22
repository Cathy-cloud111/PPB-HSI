import copy
import unittest

import run_capacity_experiments as provenance
import run_persistent_ablation as runner
import main_persistent_controls as entry
import persistent_control_audit as audit


class PersistentAblationTests(unittest.TestCase):
    def test_exact_thirty_unique_jobs(self):
        jobs = provenance.jobs_for(provenance.DATASETS, provenance.SEEDS, runner.VARIANTS)
        self.assertEqual(len(jobs), 30)
        self.assertEqual(len({job["name"] for job in jobs}), 30)
        self.assertEqual(set(runner.SUPPORTED_VARIANTS), set(runner.VARIANT_FLAGS))
        self.assertTrue(set(runner.VARIANTS).issubset(runner.SUPPORTED_VARIANTS))

    def test_entry_flags_match_runner(self):
        for variant, flags in runner.VARIANT_FLAGS.items():
            self.assertEqual(entry.VARIANT_FLAGS[variant], flags)

    def test_commands_use_locked_budget_and_formal_epochs(self):
        for dataset, budget in (("Houston2013", "1500"), ("PaviaU", "900")):
            for variant in runner.VARIANTS:
                job = {"dataset": dataset, "seed": 42, "variant": variant,
                       "name": f"{dataset}_{variant}_seed42"}
                command = runner.build_command(job, runner.ROOT / "unused", "cuda")
                flags = dict(zip(command[3::2], command[4::2]))
                self.assertEqual(flags["--memory_budget"], budget)
                self.assertEqual(flags["--epochs"], "80")
                self.assertEqual(flags["--persistent_variant"], variant)
                for key, value in runner.VARIANT_FLAGS[variant].items():
                    self.assertEqual(float(flags["--" + key]), float(value))

    def test_smoke_command_is_one_epoch_and_small_memory(self):
        job = {"dataset": "Houston2013", "seed": 42,
               "variant": "persistent_multiproto_kd",
               "name": "Houston2013_persistent_multiproto_kd_seed42"}
        command = runner.build_command(job, runner.ROOT / "unused", "cpu", smoke=True)
        self.assertEqual(command[-1], "--smoke")
        flags = dict(zip(command[3:-1:2], command[4:-1:2]))
        self.assertEqual(flags["--epochs"], "1")
        self.assertEqual(flags["--memory_budget"], "60")

    def test_matched_group_accepts_only_component_changes(self):
        base = vars(entry.get_args_parser().parse_args([
            "--persistent_variant", "persistent_kd", "--memory_budget", "1500"
        ]))
        requests, sessions = [], []
        for variant in runner.VARIANTS:
            args = copy.deepcopy(base)
            args.update(runner.VARIANT_FLAGS[variant])
            args["persistent_variant"] = variant
            identity = {"smoke": False, "protocol": audit.scientific_args(args),
                        "code_sha256": {}, "data_sha256": {}, "runtime": {}}
            requests.append({"job": {"dataset": "Houston2013", "seed": 42,
                                      "variant": variant},
                             "effective_args": args, "identity": identity,
                             "fingerprint": provenance.fingerprint(identity)})
            sessions.append([{"session": i, **{field: "same" for field in audit.PAIR_FIELDS}}
                             for i in (1, 2, 3)])
        self.assertTrue(audit.check_matched_group(requests, sessions)["passed"])
        altered = copy.deepcopy(requests)
        altered[1]["effective_args"]["lr"] = .123
        altered[1]["identity"]["protocol"] = audit.scientific_args(altered[1]["effective_args"])
        altered[1]["fingerprint"] = provenance.fingerprint(altered[1]["identity"])
        with self.assertRaises(ValueError):
            audit.check_matched_group(altered, sessions)


if __name__ == "__main__":
    unittest.main()
