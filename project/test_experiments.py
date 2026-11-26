"""Small offline tests for attack semantics, resume safety and aggregation."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "project"))
import experiments as evaluation


class TinyClassifier(nn.Module):
    def forward(self, x):
        value = x[:, 0, 0, 0]
        logits = torch.zeros((len(x), 9), device=x.device) - 10
        logits[:, 0] = 0.51 - value
        logits[:, 1] = -0.01
        logits[:, 8] = value - 0.51
        return logits * 20


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model = TinyClassifier().eval()
        self.x = torch.tensor([[[[0.5, 0.0], [0.5, 1.0]]], [[[0.54, 0.0], [0.5, 1.0]]]])
        self.y = torch.tensor([0, 0])

    def test_targeted_direction_success_and_clean_retention(self):
        attacked, pred, _ = evaluation.targeted_none_attack(self.model, self.x, epsilon=.04, steps=8, restarts=2, seed=7)
        self.assertEqual(pred.tolist(), [8, 8])
        self.assertTrue(torch.all(attacked[:, 0, 0, 0] >= self.x[:, 0, 0, 0]))
        self.assertTrue(evaluation.verify_attack(self.x, attacked, .04)["background_unchanged"])
        result = evaluation.targeted_metrics(np.array([0, 0]), pred.numpy(), np.array([0, 8]))
        self.assertEqual(result["target_success_count"], 2)
        self.assertEqual(result["clean_correct_defect_denominator"], 1)
        self.assertEqual(result["clean_correct_defect_success_rate"], 1)
        self.assertEqual(result["baseline_clean_defect_to_none_rate"], .5)

    def test_zero_epsilon_identity(self):
        attacked, pred, _ = evaluation.targeted_none_attack(self.model, self.x, epsilon=0, steps=2, restarts=2)
        self.assertTrue(torch.equal(self.x, attacked))
        self.assertTrue(torch.equal(pred, self.model(self.x).argmax(1)))

    def test_untargeted_shared_starts_monotonic(self):
        _, short, _ = evaluation.attack_batch(self.model, self.x, self.y, .04, "pgd", steps=2, restarts=2, seed=8)
        _, longer, _ = evaluation.attack_batch(self.model, self.x, self.y, .04, "pgd", steps=8, restarts=2, seed=8)
        self.assertFalse(bool(((short != self.y) & (longer == self.y)).any()))

    def test_invariants_reject_background_and_excess(self):
        invalid = self.x.clone()
        invalid[:, 0, 0, 1] = .01
        with self.assertRaises(ValueError):
            evaluation.verify_attack(self.x, invalid, .04)
        with self.assertRaises(ValueError):
            evaluation.verify_attack(self.x, self.x + .1, .04)

    def test_targeted_does_not_count_other_wrong_class(self):
        result = evaluation.targeted_metrics(np.array([0, 0]), np.array([1, 8]), np.array([0, 0]))
        self.assertEqual(result["target_success_rate"], .5)
        self.assertEqual(result["clean_correct_defect_success_rate"], .5)
        with self.assertRaises(ValueError):
            evaluation.targeted_metrics(np.array([8]), np.array([8]), np.array([8]))

    def test_protocol_mismatch_and_atomic_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            protocol = {"protocol_sha256": "abc", "definition": {"scope": "smoke"}}
            first = evaluation.lock_protocol(output, protocol)
            self.assertEqual(evaluation.lock_protocol(output, protocol), first)
            with self.assertRaises(ValueError):
                evaluation.lock_protocol(output, {"protocol_sha256": "def", "definition": {"scope": "full_test"}})
            self.assertEqual(json.loads((output / "protocol.json").read_text()), first)
            self.assertFalse(list(output.glob("*.tmp")))

    def test_batch_resume_rejects_wrong_indices_or_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "batch.npz"
            indices, labels = np.array([3, 5]), np.array([0, 1])
            evaluation.write_npz(path, config_sha256=np.array("abc"), cache_index=indices, true_class_id=labels,
                          predicted_id=labels, clean_predicted_id=labels, confidence=np.array([.6, .7]),
                          max_linf=np.array(0), background_unchanged=np.array(True), seconds=np.array(.1))
            self.assertEqual(evaluation.load_batch(path, "abc", indices, labels)["predicted_id"].tolist(), [0, 1])
            with self.assertRaises(ValueError):
                evaluation.load_batch(path, "def", indices, labels)
            with self.assertRaises(ValueError):
                evaluation.load_batch(path, "abc", indices[::-1], labels)

    def test_partial_seeds_not_aggregated(self):
        entries = []
        for seed in [77, 78, 79]:
            metrics = {"accuracy": seed / 100, "macro_f1": .5, "defect_to_none_rate": .2,
                       "per_class_recall": {name: .5 for name in evaluation.CLASS_NAMES}}
            entries.append({"method": "clean", "seed": seed, "condition": "clean", "n": 10, "metrics": metrics})
        self.assertEqual(evaluation.aggregate_runs(entries[:2]), [])
        result = evaluation.aggregate_runs(entries)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["metrics"]["accuracy"]["mean"], .78)
        self.assertAlmostEqual(result[0]["metrics"]["accuracy"]["sd"], .01)

    def test_interrupted_condition_resumes_saved_batch(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            labels = np.zeros(4, dtype=np.int64)
            images = np.ones((4, 2, 2), dtype=np.uint8)
            images[:, 0, 1] = 0
            run = {"method": "clean", "seed": 77, "checkpoint_sha256": "fixture"}
            protocol = {"protocol_sha256": "fixture", "definition": {
                "scope": "smoke", "batch_size": 2, "pgd20_steps": 2, "pgd50_steps": 2, "targeted_steps": 2,
                "restarts": 1, "runs": [run], "data_sha256": {}, "code_sha256": {}, "aggregation": "fixture",
                "smoke_warning": "fixture only"}}
            original = evaluation.attack_batch
            def interrupted(*args, **kwargs):
                if kwargs.get("seed") == 7781002:
                    raise RuntimeError("simulated interruption")
                return original(*args, **kwargs)
            with mock.patch.object(evaluation, "attack_batch", side_effect=interrupted), mock.patch.object(evaluation, "progress"):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    evaluation.evaluate_condition(output, protocol, run, "pgd20", self.model, images, labels, np.arange(4), torch.device("cpu"))
            first = output / "runs/clean_seed77/pgd20/batches/batch_0000000.npz"
            before = evaluation.sha256(first)
            self.assertFalse((first.parent.parent / "metrics.json").exists())
            with mock.patch.object(evaluation, "progress"):
                evaluation.evaluate_condition(output, protocol, run, "pgd20", self.model, images, labels, np.arange(4), torch.device("cpu"))
            self.assertEqual(before, evaluation.sha256(first))
            result = evaluation.read_json(first.parent.parent / "metrics.json")
            self.assertEqual(result["metrics"]["n"], 4)
            self.assertEqual(len(result["batch_sha256"]), 2)


if __name__ == "__main__":
    unittest.main()
