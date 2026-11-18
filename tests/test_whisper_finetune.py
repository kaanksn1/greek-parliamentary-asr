"""Tests for the Whisper fine-tuning contribution code.

Uses the real pinned tokenizer/processor files for openai/whisper-small
(text/JSON only, fetched once and cached by huggingface_hub; no safetensors
weights) so label/prefix/padding behaviour is tested against the real
tokenizer. The trainable model itself is a tiny, randomly-initialized
WhisperForConditionalGeneration sharing that tokenizer's vocabulary, injected
via the same ``model_loader`` seam greek_asr.infer.run already uses for
pretrained adapters -- this keeps the optimizer step real (real nn.Module,
real backward/step) without downloading or training the full checkpoint in
this unit test. The three real train-pilot clips under data/raw/pilot/ are
used as the only real audio fixture.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from greek_asr.models import whisper_finetune

try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
    import huggingface_hub  # noqa: F401
    _DEPS_AVAILABLE = True
except ImportError:
    _DEPS_AVAILABLE = False

_SKIP_REASON = "requires the .venv-sequence environment (torch/transformers/huggingface_hub); see requirements-sequence.txt"

ROOT = Path(__file__).resolve().parents[1]
REVISION = "973afd24965f72e36ca33b3055d56a652f456b4d"

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


def _snapshot(model_id: str, revision: str, allow_patterns: list[str]) -> str:
    """Fetch pinned files, skipping (not erroring) the test if the Hub is unreachable here."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError, OfflineModeIsEnabled
    from requests.exceptions import ConnectionError as RequestsConnectionError

    try:
        return snapshot_download(repo_id=model_id, revision=revision, token=False, allow_patterns=allow_patterns)
    except (RequestsConnectionError, HfHubHTTPError, OfflineModeIsEnabled) as error:
        raise unittest.SkipTest(f"Hugging Face Hub unreachable in this environment: {error}")


def _real_processor():
    from transformers import WhisperProcessor

    local = _snapshot("openai/whisper-small", REVISION, ["*.json", "*.txt"])
    return WhisperProcessor.from_pretrained(local, local_files_only=True)


def _tiny_model_loader(model_id: str, revision: str, *, device: str):
    """Real pinned tokenizer + a tiny randomly-initialized model sharing its vocabulary."""
    from transformers import GenerationConfig, WhisperConfig, WhisperForConditionalGeneration

    assert (model_id, revision) == ("openai/whisper-small", REVISION)
    processor = _real_processor()
    local = _snapshot(model_id, revision, ["*.json", "*.txt"])
    tokenizer = processor.tokenizer
    config = WhisperConfig(
        vocab_size=len(tokenizer), num_mel_bins=80, encoder_layers=2, decoder_layers=2,
        d_model=32, encoder_attention_heads=2, decoder_attention_heads=2,
        encoder_ffn_dim=64, decoder_ffn_dim=64, max_source_positions=1500, max_target_positions=448,
        decoder_start_token_id=tokenizer.convert_tokens_to_ids("<|startoftranscript|>"),
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.bos_token_id,
    )
    model = WhisperForConditionalGeneration(config).to(device)
    model.generation_config = GenerationConfig.from_pretrained(local, local_files_only=True)
    return processor, model


@unittest.skipUnless(_DEPS_AVAILABLE, _SKIP_REASON)
class BuildLabelsTests(unittest.TestCase):
    def test_prefix_tokens_are_real_content_and_only_padding_is_masked(self):
        processor = _real_processor()
        labels = whisper_finetune.build_labels(
            processor, ["εμείς μπορούμε σήμερα", "σε"], language="el", task="transcribe")
        tokenizer = processor.tokenizer
        prefix_ids = tokenizer.convert_tokens_to_ids(["<|startoftranscript|>", "<|el|>", "<|transcribe|>", "<|notimestamps|>"])
        for row in labels.tolist():
            self.assertEqual(row[:4], prefix_ids)  # BOS/language/task tokens are real, never masked
            # Exactly one real EOS survives per row; only true padding is -100.
            kept = [token for token in row if token != -100]
            self.assertEqual(kept.count(tokenizer.eos_token_id), 1)
        shorter, longer = labels.tolist()
        self.assertGreater(longer.count(-100), shorter.count(-100))  # the shorter "σε" row is padded more

    def test_vocabulary_is_shared_with_pinned_revision_lock(self):
        with open(ROOT / "configs/model_sources.lock.json", encoding="utf-8-sig") as handle:
            lock = json.load(handle)
        whisper_entry = next(row for row in lock["models"] if row["model_id"] == "openai/whisper-small")
        self.assertEqual(whisper_entry["revision"], REVISION)


class MatchedConditionConfigTests(unittest.TestCase):
    def test_both_whisper_conditions_share_everything_except_the_declared_data_difference(self):
        only = json.loads((ROOT / "configs/sequence/whisper_hparl_only.json").read_text("utf-8"))
        css10 = json.loads((ROOT / "configs/sequence/whisper_hparl_css10.json").read_text("utf-8"))
        allowed_to_differ = {"_comment", "condition", "train_manifest", "train_references"}
        for key in set(only) | set(css10):
            if key not in allowed_to_differ:
                self.assertEqual(only.get(key), css10.get(key), f"{key} must match for a fair data-only comparison")
        self.assertNotEqual(only["train_manifest"], css10["train_manifest"])
        self.assertEqual(only["dev_manifest"], css10["dev_manifest"])  # dev/scoring stay shared

    def test_shipped_configs_require_a_measured_step_budget_before_training(self):
        for name in ("attention.json", "transducer.json", "whisper_hparl_only.json", "whisper_hparl_css10.json"):
            config = json.loads((ROOT / "configs/sequence" / name).read_text("utf-8"))
            self.assertIsNone(config["max_steps"], f"{name}: max_steps must stay unset until Serdar's pilot measures it")


@unittest.skipUnless(_DEPS_AVAILABLE, _SKIP_REASON)
class RevisionValidationTests(unittest.TestCase):
    def test_mismatched_revision_is_rejected_before_any_download(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "model_sources.lock.json"):
                whisper_finetune.train(
                    {"model_id": "openai/whisper-small", "revision": "0" * 40, "train_manifest": "x",
                     "train_references": "x", "dev_manifest": "x", "dev_references": "x", "seed": 1,
                     "learning_rate": 1e-4, "micro_batch_size": 1, "gradient_accumulation_steps": 1,
                     "max_steps": 1, "warmup_steps": 1},
                    project_root=ROOT, output_dir=Path(directory) / "out")


@unittest.skipUnless(_DEPS_AVAILABLE, _SKIP_REASON)
class TrainAndPredictWithTinyModelTests(unittest.TestCase):
    def test_matched_budget_train_then_predict_never_leaks_gold(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            train_manifest, train_references = _write_manifest_and_references(directory, PILOT[:2], "train")
            dev_manifest, dev_references = _write_manifest_and_references(directory, PILOT[2:], "dev")
            output_dir = directory / "run"
            config = {
                "model_id": "openai/whisper-small", "revision": REVISION,
                "train_manifest": str(train_manifest), "train_references": str(train_references),
                "dev_manifest": str(dev_manifest), "dev_references": str(dev_references),
                "seed": 42, "learning_rate": 1e-3, "micro_batch_size": 1,
                "gradient_accumulation_steps": 2, "max_steps": 2, "warmup_steps": 1, "dev_eval_every": 1,
            }
            report = whisper_finetune.train(config, project_root=ROOT, output_dir=output_dir,
                                            model_loader=_tiny_model_loader)
            self.assertEqual(report["optimizer_updates_completed"], 2)
            self.assertEqual(report["effective_batch_size"], 2)
            self.assertIsNotNone(report["best_dev_loss"])
            self.assertTrue((output_dir / "checkpoint.best.pt").is_file())
            self.assertTrue((output_dir / "checkpoint.last.pt").is_file())

            predict_dir = output_dir / "predict"
            predict_report = whisper_finetune.predict(
                output_dir / "checkpoint.best.pt", dev_manifest, project_root=ROOT, split="dev",
                output_dir=predict_dir, model_loader=_tiny_model_loader)
            self.assertFalse(predict_report["references_used_as_model_input"])
            predictions = [json.loads(line) for line in (predict_dir / "predictions.jsonl").read_text("utf-8").splitlines()]
            self.assertEqual({row["id"] for row in predictions}, {PILOT[2]["id"]})
            self.assertTrue(all(isinstance(row["text"], str) for row in predictions))

    def test_predict_on_test_split_is_blocked_without_a_committed_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            manifest, _ = _write_manifest_and_references(directory, PILOT[:1], "test")
            with self.assertRaisesRegex(ValueError, "--allow-test"):
                whisper_finetune.predict(Path("unused.pt"), manifest, project_root=ROOT, split="test",
                                         output_dir=directory / "out", model_loader=_tiny_model_loader)


if __name__ == "__main__":
    unittest.main()
