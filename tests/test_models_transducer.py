"""Tests for the standard RNN-T transducer ASR model.

Uses tiny synthetic tensors for the loss/backward checks and the three real
train-pilot clips under data/raw/pilot/ for the end-to-end train->predict
check. These pilot clips are a code-path fixture, not a claim about completed
real training, dev selection or benchmark results.
"""
from __future__ import annotations

import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from greek_asr.models import common, transducer

try:
    import torch  # noqa: F401
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

_SKIP_REASON = "requires the .venv-sequence environment (torch/torchaudio); see requirements-sequence.txt"

ROOT = Path(__file__).resolve().parents[1]

PILOT = [
    {"id": "hparl:alekshs_tsipras-olomeleia_37_00240000_00243000_4_1",
     "original_id": "alekshs_tsipras-olomeleia_37_00240000_00243000_4_1",
     "audio_path": "data/raw/pilot/audio/train-pilot-0001.wav",
     "audio_sha256": "3a3d5d3dd14390eee2ec5a71be272154b6fa953766f4f746d086739e5505105c",
     "num_frames": 40160, "duration_seconds": 2.51, "reference": "εμείς μπορούμε σήμερα"},
    {"id": "hparl:dhmhtrios_gakhs-olomeleia_15_01512500_01515500_3_1",
     "original_id": "dhmhtrios_gakhs-olomeleia_15_01512500_01515500_3_1",
     "audio_path": "data/raw/pilot/audio/train-pilot-0002.wav",
     "audio_sha256": "558c1a270332a49d97d3e8993a307499fd7b719c05e6d9d6493ce2dde77896b5",
     "num_frames": 35200, "duration_seconds": 2.2, "reference": "σε αυτήν την οικονομική δράση"},
    {"id": "hparl:iwannhs_balafas-olomeleia_47_02640000_02643000_1_2",
     "original_id": "iwannhs_balafas-olomeleia_47_02640000_02643000_1_2",
     "audio_path": "data/raw/pilot/audio/train-pilot-0003.wav",
     "audio_sha256": "719b0a4fb9f5205d8cab4fc74904fe796da07ebe30ae242ccefd76955982dbe6",
     "num_frames": 30400, "duration_seconds": 1.9,
     "reference": "τους ξεφεύγει η αντικατάσταση της <spoken_noise>"},
]


def _write_manifest_and_references(directory: Path, rows, split: str) -> tuple[Path, Path]:
    manifest_path = directory / f"{split}_manifest.jsonl"
    references_path = directory / f"{split}_references.jsonl"
    with manifest_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps({
                "id": row["id"], "source": "hparl", "original_id": row["original_id"],
                "source_split": split, "split": split, "audio_path": row["audio_path"],
                "audio_sha256": row["audio_sha256"], "sampling_rate": 16000, "channels": 1,
                "num_frames": row["num_frames"], "duration_seconds": row["duration_seconds"],
            }, ensure_ascii=False) + "\n")
    with references_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps({"id": row["id"], "reference": row["reference"]}, ensure_ascii=False) + "\n")
    return manifest_path, references_path


@unittest.skipUnless(_TORCH_AVAILABLE, _SKIP_REASON)
class SyntheticLossBackwardTests(unittest.TestCase):
    def test_tiny_batch_loss_is_finite_and_backward_updates_parameters(self):
        torch, nn, rnnt_loss, TransducerASRModel = transducer._build_torch()
        torch.manual_seed(0)
        config = transducer.TransducerConfig(vocab_size=8, input_dim=6, encoder_hidden=4, encoder_layers=2,
                                             predictor_embedding=5, predictor_hidden=4, predictor_layers=1,
                                             joint_dim=6)
        model = TransducerASRModel(config)
        before = {name: parameter.clone() for name, parameter in model.named_parameters()}

        features, feature_lengths = common.pad_sequences([torch.randn(9, 6), torch.randn(5, 6)])
        start_id, blank_id = 1, 0
        predictor_inputs, _ = common.pad_sequences(
            [torch.tensor([start_id, 3, 4, 5]), torch.tensor([start_id, 6])])
        targets, target_lengths = common.pad_sequences([torch.tensor([3, 4, 5]), torch.tensor([6])])
        batch = {"features": features, "feature_lengths": feature_lengths,
                "predictor_inputs": predictor_inputs, "targets": targets, "target_lengths": target_lengths}

        loss = transducer.compute_loss(torch, rnnt_loss, model, batch, blank_id)
        self.assertTrue(torch.isfinite(loss))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())
        optimizer.step()
        changed = any(not torch.equal(before[name], parameter) for name, parameter in model.named_parameters())
        self.assertTrue(changed)

    def test_blank_id_and_lengths_contract_rejects_mismatched_shapes(self):
        torch, nn, rnnt_loss, TransducerASRModel = transducer._build_torch()
        config = transducer.TransducerConfig(vocab_size=6, input_dim=4, encoder_hidden=3, encoder_layers=1,
                                             predictor_embedding=4, predictor_hidden=3, predictor_layers=1,
                                             joint_dim=4)
        model = TransducerASRModel(config)
        features, feature_lengths = common.pad_sequences([torch.randn(4, 4)])
        predictor_inputs, _ = common.pad_sequences([torch.tensor([0, 2])])
        # target_lengths deliberately too long for the given target tensor.
        targets = torch.tensor([[2]], dtype=torch.long)
        bad_target_lengths = torch.tensor([5], dtype=torch.long)
        batch = {"features": features, "feature_lengths": feature_lengths, "predictor_inputs": predictor_inputs,
                "targets": targets, "target_lengths": bad_target_lengths}
        with self.assertRaises(Exception):
            transducer.compute_loss(torch, rnnt_loss, model, batch, blank_id=0)


@unittest.skipUnless(_TORCH_AVAILABLE, _SKIP_REASON)
class TrainAndPredictOnRealPilotAudioTests(unittest.TestCase):
    def test_train_writes_checkpoints_then_predict_never_leaks_gold_and_covers_all_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            train_manifest, train_references = _write_manifest_and_references(directory, PILOT[:2], "train")
            dev_manifest, dev_references = _write_manifest_and_references(directory, PILOT[2:], "dev")
            output_dir = directory / "run"
            config = {
                "train_manifest": str(train_manifest), "train_references": str(train_references),
                "dev_manifest": str(dev_manifest), "dev_references": str(dev_references),
                "seed": 42, "learning_rate": 0.01, "batch_size": 2, "max_steps": 2,
                "dev_eval_every": 1, "sample_rate": 16000,
            }
            report = transducer.train(config, project_root=ROOT, output_dir=output_dir)
            self.assertEqual(report["steps_completed"], 2)
            self.assertIsNotNone(report["best_dev_loss"])
            self.assertTrue((output_dir / "checkpoint.best.pt").is_file())
            self.assertTrue((output_dir / "checkpoint.last.pt").is_file())

            checkpoint = common.load_checkpoint(output_dir / "checkpoint.best.pt")
            vocab = common.Vocabulary.from_json(checkpoint["vocab"])
            self.assertNotIn("ξ", vocab.stoi)  # only in the dev-only reference

            predict_dir = output_dir / "predict"
            predict_report = transducer.predict(output_dir / "checkpoint.best.pt", dev_manifest,
                                                project_root=ROOT, split="dev", output_dir=predict_dir)
            self.assertFalse(predict_report["references_used_as_model_input"])
            predictions = [json.loads(line) for line in (predict_dir / "predictions.jsonl").read_text("utf-8").splitlines()]
            self.assertEqual({row["id"] for row in predictions}, {PILOT[2]["id"]})
            self.assertTrue(all(isinstance(row["text"], str) for row in predictions))

    def test_shipped_config_refuses_an_unmeasured_step_budget(self):
        config = json.loads((ROOT / "configs/sequence/transducer.json").read_text("utf-8"))
        with self.assertRaisesRegex(ValueError, "measured pilot run"):
            transducer.train(config, project_root=ROOT, output_dir=ROOT / "__should_not_be_created__")

    def test_predict_on_test_split_is_blocked_without_a_committed_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            manifest, _ = _write_manifest_and_references(directory, PILOT[:1], "test")
            with self.assertRaisesRegex(ValueError, "--allow-test"):
                transducer.predict(Path("unused.pt"), manifest, project_root=ROOT, split="test",
                                   output_dir=directory / "out")

    def _train_small_model(self, directory: Path):
        train_manifest, train_references = _write_manifest_and_references(directory, PILOT[:2], "train")
        dev_manifest, dev_references = _write_manifest_and_references(directory, PILOT[2:], "dev")
        config = {
            "train_manifest": str(train_manifest), "train_references": str(train_references),
            "dev_manifest": str(dev_manifest), "dev_references": str(dev_references),
            "seed": 42, "learning_rate": 0.01, "batch_size": 2, "max_steps": 2,
            "dev_eval_every": 1, "sample_rate": 16000,
        }
        report = transducer.train(config, project_root=ROOT, output_dir=directory / "run")
        return report, directory / "run", dev_manifest

    def test_train_report_records_full_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            report, _, _ = self._train_small_model(Path(directory))
        self.assertEqual(report["code_commit"], common.git_commit_hash(ROOT))
        self.assertIn("transducer.py", report["code_sha256"])
        self.assertIn("common.py", report["code_sha256"])
        for field in ("train_manifest_sha256", "train_references_sha256", "dev_manifest_sha256",
                      "dev_references_sha256"):
            self.assertRegex(report[field], r"^[0-9a-f]{64}$")
        self.assertEqual(report["effective_batch_size"], 2)
        self.assertEqual(report["precision"], "float32")
        self.assertGreater(report["trainable_parameters"], 0)
        self.assertGreaterEqual(report["training_seconds"], 0)

    def test_checkpoint_save_then_load_gives_identical_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            _, run_dir, dev_manifest = self._train_small_model(directory)
            checkpoint_path = run_dir / "checkpoint.best.pt"

            def predict_texts(output_name: str) -> list[str]:
                transducer.predict(checkpoint_path, dev_manifest, project_root=ROOT, split="dev",
                                   output_dir=run_dir / output_name)
                lines = [json.loads(line) for line in (run_dir / output_name / "predictions.jsonl").read_text("utf-8").splitlines()]
                return [row["text"] for row in sorted(lines, key=lambda row: row["id"])]

            first = predict_texts("predict_first")
            second = predict_texts("predict_second")
        self.assertEqual(first, second)

    def test_rerunning_into_an_existing_output_dir_is_refused_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            train_manifest, train_references = _write_manifest_and_references(directory, PILOT[:2], "train")
            dev_manifest, dev_references = _write_manifest_and_references(directory, PILOT[2:], "dev")
            config = {
                "train_manifest": str(train_manifest), "train_references": str(train_references),
                "dev_manifest": str(dev_manifest), "dev_references": str(dev_references),
                "seed": 42, "learning_rate": 0.01, "batch_size": 2, "max_steps": 1,
                "dev_eval_every": 1, "sample_rate": 16000,
            }
            output_dir = directory / "run"
            transducer.train(config, project_root=ROOT, output_dir=output_dir)
            before = (output_dir / "checkpoint.last.pt").read_bytes()
            with self.assertRaises(FileExistsError):
                transducer.train(config, project_root=ROOT, output_dir=output_dir)
            self.assertEqual((output_dir / "checkpoint.last.pt").read_bytes(), before)

    def test_a_failure_mid_prediction_leaves_a_partial_file_and_a_failed_run_json(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            _, run_dir, dev_manifest = self._train_small_model(directory)
            predict_dir = run_dir / "predict_failure"
            original = common.load_waveform
            calls = {"count": 0}

            def flaky_load_waveform(*args, **kwargs):
                calls["count"] += 1
                if calls["count"] > 1:
                    raise RuntimeError("simulated decode failure")
                return original(*args, **kwargs)

            train_manifest, _ = _write_manifest_and_references(directory, PILOT[:2], "train")
            with unittest.mock.patch.object(common, "load_waveform", side_effect=flaky_load_waveform):
                with self.assertRaises(RuntimeError):
                    transducer.predict(run_dir / "checkpoint.best.pt", train_manifest, project_root=ROOT,
                                       split="train", output_dir=predict_dir)
            self.assertTrue((predict_dir / "predictions.partial.jsonl").is_file())
            self.assertFalse((predict_dir / "predictions.jsonl").exists())
            run_report = json.loads((predict_dir / "run.json").read_text("utf-8"))
            self.assertEqual(run_report["status"], "failed")
            self.assertEqual(run_report["completed_records"], 1)


if __name__ == "__main__":
    unittest.main()
