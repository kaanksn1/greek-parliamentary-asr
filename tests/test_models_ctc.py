"""Tests for Kemal's plain CTC baseline and the shared CTC helpers.

Pure decoding/feasibility checks use tiny synthetic tensors. End-to-end
train -> checkpoint -> predict runs use the three committed real train-pilot
clips as a code-path fixture; they are not training results.
"""
from __future__ import annotations

import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from greek_asr.models import ctc_utils

try:
    import torch

    from greek_asr.models import common, ctc
    _TORCH = True
except ImportError:
    _TORCH = False

from tests.test_models_attention import PILOT, _write_manifest_and_references

ROOT = Path(__file__).resolve().parents[1]
SKIP = "requires torch/torchaudio"
CLEAN_PILOT = [dict(row, reference=ctc_utils.strip_annotation_tags(row["reference"])) for row in PILOT]


class CtcHelperTests(unittest.TestCase):
    def test_collapse_merges_repeats_before_dropping_blanks(self):
        blank = 0
        self.assertEqual(ctc_utils.ctc_collapse([1, 1, 0, 1, 2, 2, 0, 0], blank), [1, 1, 2])
        self.assertEqual(ctc_utils.ctc_collapse([0, 0, 0], blank), [])
        self.assertEqual(ctc_utils.ctc_collapse([3, 3, 3], blank), [3])

    def test_required_frames_count_a_blank_between_repeats(self):
        self.assertEqual(ctc_utils.ctc_required_frames([5, 5, 6]), 4)
        self.assertEqual(ctc_utils.ctc_required_frames([5, 6, 5]), 3)
        self.assertEqual(ctc_utils.ctc_required_frames([]), 0)

    def test_feasibility_policies(self):
        items = [("ok", 10, [1, 2, 3]), ("short", 3, [1, 1, 2]), ("empty", 9, [])]
        with self.assertRaises(ValueError):
            ctc_utils.check_ctc_feasibility(items, policy="error", max_excluded_fraction=1.0)
        excluded, report = ctc_utils.check_ctc_feasibility(items, policy="exclude_and_report",
                                                           max_excluded_fraction=1.0)
        self.assertEqual(excluded, {"short", "empty"})
        self.assertEqual({row["reason"] for row in report}, {"too_few_output_frames", "empty_target"})
        with self.assertRaises(ValueError):
            ctc_utils.check_ctc_feasibility(items, policy="exclude_and_report", max_excluded_fraction=0.5)
        with self.assertRaises(ValueError):
            ctc_utils.check_ctc_feasibility(items, policy="drop_silently", max_excluded_fraction=1.0)

    def test_annotation_tags_are_removed_in_memory_only(self):
        self.assertEqual(ctc_utils.strip_annotation_tags("της <spoken_noise>  και"), "της και")
        self.assertEqual(ctc_utils.strip_annotation_tags("<spoken_noise>"), "")

    def test_measurement_configs_differ_from_full_configs_only_in_budget(self):
        budget_keys = {"_comment", "max_steps", "dev_eval_every", "warmup_steps", "freeze_transformer_updates"}
        for name in ("ctc", "hubert_ctc"):
            full = json.loads((ROOT / f"configs/kemal/{name}.json").read_text(encoding="utf-8"))
            measure = json.loads((ROOT / f"configs/kemal/{name}_measure.json").read_text(encoding="utf-8"))
            self.assertIsInstance(full["max_steps"], int)  # measured owner budget is committed
            self.assertGreater(full["max_steps"], 0)
            self.assertIsInstance(measure["max_steps"], int)
            self.assertEqual({k: v for k, v in full.items() if k not in budget_keys},
                             {k: v for k, v in measure.items() if k not in budget_keys})

    def test_bucketed_batches_cover_every_index_once(self):
        lengths = [float(i % 7) for i in range(53)]
        batches = ctc_utils.length_bucketed_batches(lengths, 4, seed=3, bucket_batches=2)
        flat = [index for batch in batches for index in batch]
        self.assertEqual(sorted(flat), list(range(53)))
        self.assertEqual(batches, ctc_utils.length_bucketed_batches(lengths, 4, seed=3, bucket_batches=2))


@unittest.skipUnless(_TORCH, SKIP)
class CtcDecodingTests(unittest.TestCase):
    def test_greedy_decode_ignores_frames_beyond_valid_length(self):
        vocab = 4
        best_paths = [[1, 1, 0, 1, 2, 3, 3], [2, 0, 2, 3, 3, 3, 3]]
        log_probs = torch.full((2, 7, vocab), -10.0)
        for b, path in enumerate(best_paths):
            for t, token in enumerate(path):
                log_probs[b, t, token] = 0.0
        decoded = ctc_utils.ctc_greedy_decode(log_probs, torch.tensor([5, 3]), blank_id=0)
        self.assertEqual(decoded, [[1, 1, 2], [2, 2]])
        with self.assertRaises(ValueError):
            ctc_utils.ctc_greedy_decode(log_probs, torch.tensor([8, 3]), blank_id=0)

    def test_decode_ids_keeps_genuine_empty_output_and_drops_unknown(self):
        vocab = common.build_char_vocabulary(["αβ γ"], ctc.SPECIALS)
        unk, alpha = vocab.stoi[common.UNK], vocab.stoi["α"]
        self.assertEqual(ctc.decode_ids(vocab, []), "")
        self.assertEqual(ctc.decode_ids(vocab, [alpha, unk, alpha]), "αα")


@unittest.skipUnless(_TORCH, SKIP)
class CtcModelTests(unittest.TestCase):
    def setUp(self):
        self.torch, CTCModel = ctc._build_torch()
        torch.manual_seed(0)
        self.config = ctc.CTCConfig(vocab_size=6, input_dim=16, conv_channels=3, model_dim=8, lstm_hidden=5,
                                    lstm_layers=2, dropout=0.0)
        self.model = CTCModel(self.config)

    def test_output_length_matches_formula(self):
        for length in (1, 2, 5, 9, 16, 17):
            _, out = self.model(torch.randn(1, length, 16), torch.tensor([length]))
            self.assertEqual(int(out), int(ctc.subsampled_length(torch.tensor(length))))

    def test_padding_does_not_change_an_utterance_output(self):
        self.model.eval()
        short, long = torch.randn(9, 16), torch.randn(23, 16)
        alone, alone_length = self.model(short.unsqueeze(0), torch.tensor([9]))
        features, lengths = common.pad_sequences([short, long])
        batched, out_lengths = self.model(features, lengths)
        n = int(alone_length)
        self.assertEqual(n, int(out_lengths[0]))
        self.assertTrue(torch.allclose(alone[0, :n], batched[0, :n], atol=1e-5))

    def test_forward_backward_updates_parameters_with_finite_gradients(self):
        features, lengths = common.pad_sequences([torch.randn(40, 16), torch.randn(28, 16)])
        batch = {"ids": ["a", "b"], "features": features, "lengths": lengths,
                 "targets": torch.tensor([1, 2, 2, 3, 4, 1]), "target_lengths": torch.tensor([4, 2])}
        before = {name: p.detach().clone() for name, p in self.model.named_parameters()}
        loss = ctc.ctc_loss(self.model, batch, blank_id=0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for parameter in self.model.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())
        torch.optim.SGD(self.model.parameters(), lr=0.1).step()
        changed = [name for name, p in self.model.named_parameters() if not torch.equal(p, before[name])]
        self.assertTrue(changed)

    def test_impossible_alignment_is_an_error_not_zero_loss(self):
        features, lengths = common.pad_sequences([torch.randn(8, 16)])  # 2 output frames
        batch = {"ids": ["too-long"], "features": features, "lengths": lengths,
                 "targets": torch.tensor([1, 2, 3, 4]), "target_lengths": torch.tensor([4])}
        with self.assertRaises(RuntimeError):
            ctc.ctc_loss(self.model, batch, blank_id=0)


@unittest.skipUnless(_TORCH, SKIP)
class CtcPipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.train_manifest, self.train_refs = _write_manifest_and_references(self.dir, CLEAN_PILOT, "train")
        self.dev_manifest, self.dev_refs = _write_manifest_and_references(self.dir, PILOT, "dev")
        self.config = {
            "train_manifest": str(self.train_manifest), "train_references": str(self.train_refs),
            "dev_manifest": str(self.dev_manifest), "dev_references": str(self.dev_refs), "seed": 7,
            "learning_rate": 0.003, "batch_size": 2, "max_steps": 2, "dev_eval_every": 1, "warmup_steps": 1,
            "model": {"conv_channels": 3, "model_dim": 8, "lstm_hidden": 6, "lstm_layers": 1, "dropout": 0.0},
            "spec_augment": {"freq_masks": 1, "freq_width": 5, "time_masks": 1, "time_ratio": 0.05},
            "cmvn_utterances": 3,
        }

    def tearDown(self):
        self.tmp.cleanup()

    def test_train_checkpoint_reload_and_predict_every_id(self):
        report = ctc.train(self.config, project_root=ROOT, output_dir=self.dir / "run")
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["optimizer_updates_completed"], 2)
        self.assertEqual(report["train_items"], 3)
        self.assertIsNotNone(report["best_optimizer_update"])
        self.assertEqual(len(report["dev_history"]), 2)
        self.assertGreater(report["seconds_per_update_excluding_dev"], 0.0)
        checkpoint = self.dir / "run/checkpoint.best.pt"
        first = ctc.predict(checkpoint, self.dev_manifest, project_root=ROOT, split="dev",
                            output_dir=self.dir / "pred1", batch_size=2)
        second = ctc.predict(checkpoint, self.dev_manifest, project_root=ROOT, split="dev",
                             output_dir=self.dir / "pred2", batch_size=3)
        self.assertEqual(first["status"], "complete")
        rows1 = [json.loads(line) for line in (self.dir / "pred1/predictions.jsonl").read_text().splitlines()]
        rows2 = [json.loads(line) for line in (self.dir / "pred2/predictions.jsonl").read_text().splitlines()]
        self.assertEqual([row["id"] for row in rows1], [row["id"] for row in PILOT])
        self.assertEqual(rows1, rows2)  # reloaded checkpoint, different batch padding, same output
        self.assertFalse(first["references_used_as_model_input"])

    def test_missing_budget_is_refused(self):
        config = dict(self.config, max_steps=None)
        with self.assertRaises(ValueError):
            ctc.train(config, project_root=ROOT, output_dir=self.dir / "run")
        self.assertFalse((self.dir / "run").exists())

    def test_cli_budget_override_is_recorded(self):
        config = dict(self.config, max_steps=None)
        report = ctc.train(config, project_root=ROOT, output_dir=self.dir / "run", max_steps_override=1)
        self.assertEqual((report["max_steps"], report["max_steps_source"]), (1, "cli"))

    def test_test_split_requires_allow_flag_and_protocol(self):
        ctc.train(self.config, project_root=ROOT, output_dir=self.dir / "run")
        checkpoint = self.dir / "run/checkpoint.best.pt"
        with self.assertRaises(ValueError):
            ctc.predict(checkpoint, self.dev_manifest, project_root=ROOT, split="test", output_dir=self.dir / "t1")
        with self.assertRaises(ValueError):
            ctc.predict(checkpoint, self.dev_manifest, project_root=ROOT, split="test", output_dir=self.dir / "t2",
                        allow_test=True)
        self.assertFalse((self.dir / "t1").exists() or (self.dir / "t2").exists())

    def test_empty_prediction_is_kept_and_failure_leaves_partial_file(self):
        ctc.train(self.config, project_root=ROOT, output_dir=self.dir / "run")
        checkpoint = self.dir / "run/checkpoint.best.pt"
        with unittest.mock.patch.object(ctc.ctc_utils, "ctc_greedy_decode",
                                        side_effect=lambda logits, lengths, blank: [[] for _ in lengths]):
            ctc.predict(checkpoint, self.dev_manifest, project_root=ROOT, split="dev", output_dir=self.dir / "empty")
        rows = [json.loads(line) for line in (self.dir / "empty/predictions.jsonl").read_text().splitlines()]
        self.assertEqual([row["text"] for row in rows], ["", "", ""])

        calls = {"n": 0}
        original = ctc._Featurizer.__call__

        def fail_second(self_, path):
            calls["n"] += 1
            if calls["n"] == 2:
                raise ValueError("simulated decode failure")
            return original(self_, path)

        with unittest.mock.patch.object(ctc._Featurizer, "__call__", fail_second):
            with self.assertRaises(ValueError):
                ctc.predict(checkpoint, self.dev_manifest, project_root=ROOT, split="dev",
                            output_dir=self.dir / "failed", batch_size=1)
        self.assertTrue((self.dir / "failed/predictions.partial.jsonl").exists())
        self.assertFalse((self.dir / "failed/predictions.jsonl").exists())
        self.assertEqual(json.loads((self.dir / "failed/run.json").read_text())["status"], "failed")

    def test_manifest_reference_id_mismatch_is_refused_before_training(self):
        self.train_refs.write_text("".join(line + "\n" for line in self.train_refs.read_text().splitlines()[:2]),
                                   encoding="utf-8")
        with self.assertRaises(ValueError):
            ctc.train(self.config, project_root=ROOT, output_dir=self.dir / "run")
        self.assertFalse((self.dir / "run").exists())

    def test_a_dropped_prediction_fails_the_run_instead_of_completing(self):
        ctc.train(self.config, project_root=ROOT, output_dir=self.dir / "run")
        checkpoint = self.dir / "run/checkpoint.best.pt"
        with unittest.mock.patch.object(ctc.ctc_utils, "ctc_greedy_decode",
                                        side_effect=lambda logits, lengths, blank: [[1]] * (len(lengths) - 1)):
            with self.assertRaises(ValueError):
                ctc.predict(checkpoint, self.dev_manifest, project_root=ROOT, split="dev",
                            output_dir=self.dir / "missing", batch_size=3)
        self.assertFalse((self.dir / "missing/predictions.jsonl").exists())
        self.assertEqual(len((self.dir / "missing/predictions.partial.jsonl").read_text().splitlines()), 2)
        self.assertEqual(json.loads((self.dir / "missing/run.json").read_text())["status"], "failed")

    def test_infeasible_training_item_follows_policy(self):
        rows = [dict(CLEAN_PILOT[0], reference="α" * 400)] + CLEAN_PILOT[1:]
        bad_dir = self.dir / "bad"
        bad_dir.mkdir()
        manifest, refs = _write_manifest_and_references(bad_dir, rows, "train")
        config = dict(self.config, train_manifest=str(manifest), train_references=str(refs),
                      infeasible_policy="error")
        with self.assertRaises(ValueError):
            ctc.train(config, project_root=ROOT, output_dir=self.dir / "r1")
        config.update(infeasible_policy="exclude_and_report", max_excluded_fraction=0.5)
        report = ctc.train(config, project_root=ROOT, output_dir=self.dir / "r2")
        self.assertEqual(report["train_items"], 2)
        self.assertEqual([row["id"] for row in report["train_excluded_infeasible"]], [CLEAN_PILOT[0]["id"]])


if __name__ == "__main__":
    unittest.main()
