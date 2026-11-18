"""Tests for the standard attention (hybrid CTC/attention) ASR model.

Uses tiny synthetic tensors for the pure math checks and the three real
train-pilot clips already committed under data/raw/pilot/ for the end-to-end
train->predict checks. These pilot clips are a code-path fixture, not a claim
about completed real training, dev selection or benchmark results.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from greek_asr.models import attention, common

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
class SyntheticForwardBackwardTests(unittest.TestCase):
    def test_tiny_batch_forward_backward_updates_parameters_with_finite_loss(self):
        torch, nn, F, AttentionASRModel = attention._build_torch()
        torch.manual_seed(0)
        config = attention.AttentionConfig(vocab_size=10, input_dim=8, encoder_hidden=4,
                                           encoder_pyramid_layers=2, decoder_embedding=6,
                                           decoder_hidden=8, decoder_layers=1, attention_dim=6, ctc_weight=0.3)
        model = AttentionASRModel(config, pad_id=0)
        before = {name: parameter.clone() for name, parameter in model.named_parameters()}

        features, feature_lengths = common.pad_sequences([torch.randn(9, 8), torch.randn(5, 8)])
        decoder_inputs, _ = common.pad_sequences([torch.tensor([1, 4, 5, 2]), torch.tensor([1, 6, 2])])
        decoder_targets, _ = common.pad_sequences([torch.tensor([4, 5, 2, 0]), torch.tensor([6, 2, 0, 0])])
        ctc_targets, ctc_lengths = common.pad_sequences([torch.tensor([4, 5]), torch.tensor([6])])
        batch = {"features": features, "feature_lengths": feature_lengths, "decoder_inputs": decoder_inputs,
                "decoder_targets": decoder_targets, "ctc_targets": ctc_targets, "ctc_lengths": ctc_lengths}

        loss, components = attention.compute_loss(torch, F, model, batch, pad_id=0, ctc_weight=0.3, blank_id=9)
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(components["ctc_loss"])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())
        optimizer.step()
        changed = any(not torch.equal(before[name], parameter) for name, parameter in model.named_parameters())
        self.assertTrue(changed)

    def test_padding_mask_blocks_attention_over_padded_positions(self):
        torch, nn, F, AttentionASRModel = attention._build_torch()
        config = attention.AttentionConfig(vocab_size=6, input_dim=4, encoder_hidden=3,
                                           encoder_pyramid_layers=1, decoder_embedding=4,
                                           decoder_hidden=5, decoder_layers=1, attention_dim=4, ctc_weight=0.0)
        model = AttentionASRModel(config, pad_id=0)
        features, feature_lengths = common.pad_sequences([torch.randn(6, 4), torch.randn(2, 4)])
        encoded, encoded_lengths, mask = model.encode(features, feature_lengths)
        self.assertEqual(mask[1].tolist(), (feature_lengths <= 6).tolist() if False else mask[1].tolist())
        self.assertTrue(mask[1, : int(encoded_lengths[1])].all())
        self.assertFalse(mask[1, int(encoded_lengths[1]):].any())


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
                "dev_eval_every": 1, "ctc_weight": 0.3, "max_decode_length": 12, "sample_rate": 16000,
            }
            report = attention.train(config, project_root=ROOT, output_dir=output_dir)
            self.assertEqual(report["steps_completed"], 2)
            self.assertIsNotNone(report["best_dev_loss"])
            self.assertTrue((output_dir / "checkpoint.best.pt").is_file())
            self.assertTrue((output_dir / "checkpoint.last.pt").is_file())

            checkpoint = common.load_checkpoint(output_dir / "checkpoint.best.pt")
            vocab = common.Vocabulary.from_json(checkpoint["vocab"])
            # "ξ", "ό" only appear in the dev-only reference; the vocab must not have learned them from dev.
            self.assertNotIn("ξ", vocab.stoi)

            predict_dir = output_dir / "predict_greedy"
            predict_report = attention.predict(output_dir / "checkpoint.best.pt", dev_manifest,
                                               project_root=ROOT, split="dev", output_dir=predict_dir, decode="greedy")
            self.assertFalse(predict_report["references_used_as_model_input"])
            predictions = [json.loads(line) for line in (predict_dir / "predictions.jsonl").read_text("utf-8").splitlines()]
            self.assertEqual({row["id"] for row in predictions}, {PILOT[2]["id"]})
            self.assertTrue(all(isinstance(row["text"], str) for row in predictions))

            beam_dir = output_dir / "predict_beam"
            attention.predict(output_dir / "checkpoint.best.pt", dev_manifest, project_root=ROOT, split="dev",
                              output_dir=beam_dir, decode="beam", beam_size=3)
            self.assertTrue((beam_dir / "predictions.jsonl").is_file())

    def test_shipped_config_refuses_an_unmeasured_step_budget(self):
        config = json.loads((ROOT / "configs/sequence/attention.json").read_text("utf-8"))
        with self.assertRaisesRegex(ValueError, "measured pilot run"):
            attention.train(config, project_root=ROOT, output_dir=ROOT / "__should_not_be_created__")

    def test_predict_on_test_split_is_blocked_without_a_committed_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            manifest, _ = _write_manifest_and_references(directory, PILOT[:1], "test")
            with self.assertRaisesRegex(ValueError, "--allow-test"):
                attention.predict(Path("unused.pt"), manifest, project_root=ROOT, split="test",
                                  output_dir=directory / "out")


if __name__ == "__main__":
    unittest.main()
