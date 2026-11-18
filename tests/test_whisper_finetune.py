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
import unittest.mock
from pathlib import Path

from greek_asr.models import common, whisper_finetune

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
    def _decoder_start_token_id(self, processor):
        return processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")

    def test_prefix_tokens_are_real_content_and_only_padding_is_masked(self):
        processor = _real_processor()
        decoder_start_token_id = self._decoder_start_token_id(processor)
        labels = whisper_finetune.build_labels(
            processor, ["εμείς μπορούμε σήμερα", "σε"], language="el", task="transcribe",
            decoder_start_token_id=decoder_start_token_id)
        tokenizer = processor.tokenizer
        # The leading decoder_start_token_id is stripped from the returned labels
        # (the model's own shift_tokens_right re-adds exactly one); so row[0] is
        # now the language token, not <|startoftranscript|> again.
        prefix_ids = tokenizer.convert_tokens_to_ids(["<|el|>", "<|transcribe|>", "<|notimestamps|>"])
        for row in labels.tolist():
            self.assertEqual(row[:3], prefix_ids)
            self.assertNotEqual(row[0], decoder_start_token_id)
            # Exactly one real EOS survives per row; only true padding is -100.
            kept = [token for token in row if token != -100]
            self.assertEqual(kept.count(tokenizer.eos_token_id), 1)
        shorter, longer = labels.tolist()
        self.assertGreater(longer.count(-100), shorter.count(-100))  # the shorter "σε" row is padded more

    def test_decoder_input_has_the_start_token_exactly_once_after_the_models_own_shift(self):
        """Regression for docs/reviews/ADNAN_0437631_GPU.json "P1 - Whisper
        decoder başlangıcı iki kez ekleniyor": applying tokenization AND the
        model's real decoder shift together must produce the start token
        exactly once. Run against the pre-fix build_labels (which kept the
        leading start token), this assertion fails with the start token
        appearing twice -- that is the regression this test guards.
        """
        from transformers import WhisperConfig
        from transformers.models.whisper.modeling_whisper import shift_tokens_right

        processor = _real_processor()
        config = WhisperConfig.from_pretrained(
            _snapshot("openai/whisper-small", REVISION, ["*.json", "*.txt"]), local_files_only=True)
        decoder_start_token_id = config.decoder_start_token_id
        labels = whisper_finetune.build_labels(
            processor, ["καλημέρα"], language="el", task="transcribe",
            decoder_start_token_id=decoder_start_token_id)
        decoder_inputs = shift_tokens_right(labels, config.pad_token_id, decoder_start_token_id)
        for row in decoder_inputs.tolist():
            self.assertEqual(row.count(decoder_start_token_id), 1)
        self.assertEqual(decoder_inputs[0, 0].item(), decoder_start_token_id)
        self.assertNotEqual(decoder_inputs[0, 1].item(), decoder_start_token_id)

    def test_mismatched_decoder_start_token_id_is_rejected(self):
        processor = _real_processor()
        with self.assertRaisesRegex(ValueError, "decoder_start_token_id"):
            whisper_finetune.build_labels(processor, ["καλημέρα"], language="el", task="transcribe",
                                          decoder_start_token_id=0)

    def test_vocabulary_is_shared_with_pinned_revision_lock(self):
        with open(ROOT / "configs/model_sources.lock.json", encoding="utf-8-sig") as handle:
            lock = json.load(handle)
        whisper_entry = next(row for row in lock["models"] if row["model_id"] == "openai/whisper-small")
        self.assertEqual(whisper_entry["revision"], REVISION)


@unittest.skipUnless(_DEPS_AVAILABLE, _SKIP_REASON)
class LongAudioGuardTests(unittest.TestCase):
    """docs/reviews/ADNAN_0437631_GPU.json "P2 - uzun sesi sessiz kesme": the
    feature extractor silently truncates anything longer than 30 seconds
    instead of raising. These use synthetic arrays (the same approach the
    review itself used -- "iki sentetik 31 saniyelik girdi"), not fabricated
    corpus audio, since no real HParl/CSS10 recording in the shipped
    manifests is anywhere near this long (max 26.4s; see
    docs/SEQUENCE_MODELS.md "Known limitations").
    """

    def test_exactly_30_seconds_is_accepted(self):
        import numpy as np

        whisper_finetune._require_within_whisper_window(
            np.zeros(30 * 16000, dtype=np.float32), 16000, "synthetic:30s")

    def test_just_over_30_seconds_raises_instead_of_silently_truncating(self):
        import numpy as np

        with self.assertRaisesRegex(ValueError, "exceeds Whisper"):
            whisper_finetune._require_within_whisper_window(
                np.zeros(30 * 16000 + 1, dtype=np.float32), 16000, "synthetic:30s_plus_one_sample")


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

    def _base_config(self, directory: Path, train_manifest: Path, train_references: Path,
                     dev_manifest: Path, dev_references: Path) -> dict:
        return {
            "model_id": "openai/whisper-small", "revision": REVISION,
            "train_manifest": str(train_manifest), "train_references": str(train_references),
            "dev_manifest": str(dev_manifest), "dev_references": str(dev_references),
            "seed": 42, "learning_rate": 1e-3, "micro_batch_size": 1,
            "gradient_accumulation_steps": 2, "max_steps": 2, "warmup_steps": 1, "dev_eval_every": 1,
        }

    def _train_small_model(self, directory: Path):
        train_manifest, train_references = _write_manifest_and_references(directory, PILOT[:2], "train")
        dev_manifest, dev_references = _write_manifest_and_references(directory, PILOT[2:], "dev")
        config = self._base_config(directory, train_manifest, train_references, dev_manifest, dev_references)
        report = whisper_finetune.train(config, project_root=ROOT, output_dir=directory / "run",
                                         model_loader=_tiny_model_loader)
        return report, directory / "run", dev_manifest

    def test_train_report_records_full_provenance_and_source_breakdown(self):
        with tempfile.TemporaryDirectory() as directory:
            report, _, _ = self._train_small_model(Path(directory))
        self.assertEqual(report["code_commit"], common.git_commit_hash(ROOT))
        self.assertIn("whisper_finetune.py", report["code_sha256"])
        self.assertIn("common.py", report["code_sha256"])
        for field in ("train_manifest_sha256", "train_references_sha256", "dev_manifest_sha256",
                      "dev_references_sha256"):
            self.assertRegex(report[field], r"^[0-9a-f]{64}$")
        self.assertEqual(report["precision"], "float32")
        self.assertGreater(report["trainable_parameters"], 0)
        self.assertGreaterEqual(report["training_seconds"], 0)
        # HParl-only condition: the breakdown must show only hparl, no css10_el.
        self.assertEqual(set(report["train_source_breakdown"]), {"hparl"})
        self.assertEqual(report["train_source_breakdown"]["hparl"]["utterances"], 2)

    def test_mixed_condition_records_the_actual_css10_fraction(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            css10_row = dict(PILOT[2])
            css10_row["id"] = "css10_el:fixture-1"
            css10_row["original_id"] = "fixture-1"
            mixed_rows = PILOT[:2] + [css10_row]
            manifest_path = directory / "train_manifest.jsonl"
            references_path = directory / "train_references.jsonl"
            with manifest_path.open("w", encoding="utf-8") as stream:
                for row in PILOT[:2]:
                    stream.write(json.dumps({
                        "id": row["id"], "source": "hparl", "original_id": row["original_id"],
                        "source_split": "train", "split": "train", "audio_path": row["audio_path"],
                        "audio_sha256": row["audio_sha256"], "sampling_rate": 16000, "channels": 1,
                        "num_frames": row["num_frames"], "duration_seconds": row["duration_seconds"],
                    }, ensure_ascii=False) + "\n")
                stream.write(json.dumps({
                    "id": css10_row["id"], "source": "css10_el", "original_id": css10_row["original_id"],
                    "source_split": "unsplit", "split": "train", "audio_path": css10_row["audio_path"],
                    "audio_sha256": css10_row["audio_sha256"], "sampling_rate": 16000, "channels": 1,
                    "num_frames": css10_row["num_frames"], "duration_seconds": css10_row["duration_seconds"],
                }, ensure_ascii=False) + "\n")
            with references_path.open("w", encoding="utf-8") as stream:
                for row in mixed_rows:
                    stream.write(json.dumps({"id": row["id"], "reference": row["reference"]}, ensure_ascii=False) + "\n")
            dev_manifest, dev_references = _write_manifest_and_references(directory, PILOT[2:], "dev")
            config = self._base_config(directory, manifest_path, references_path, dev_manifest, dev_references)
            report = whisper_finetune.train(config, project_root=ROOT, output_dir=directory / "run",
                                             model_loader=_tiny_model_loader)
        self.assertEqual(set(report["train_source_breakdown"]), {"hparl", "css10_el"})
        self.assertEqual(report["train_source_breakdown"]["css10_el"]["utterances"], 1)
        self.assertEqual(report["train_source_breakdown"]["hparl"]["utterances"], 2)

    def test_checkpoint_save_then_load_gives_identical_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            _, run_dir, dev_manifest = self._train_small_model(directory)
            checkpoint_path = run_dir / "checkpoint.best.pt"

            def predict_texts(output_name: str) -> list[str]:
                whisper_finetune.predict(checkpoint_path, dev_manifest, project_root=ROOT, split="dev",
                                         output_dir=run_dir / output_name, model_loader=_tiny_model_loader)
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
            config = self._base_config(directory, train_manifest, train_references, dev_manifest, dev_references)
            output_dir = directory / "run"
            whisper_finetune.train(config, project_root=ROOT, output_dir=output_dir, model_loader=_tiny_model_loader)
            before = (output_dir / "checkpoint.last.pt").read_bytes()
            with self.assertRaises(FileExistsError):
                whisper_finetune.train(config, project_root=ROOT, output_dir=output_dir,
                                       model_loader=_tiny_model_loader)
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
                    whisper_finetune.predict(run_dir / "checkpoint.best.pt", train_manifest, project_root=ROOT,
                                             split="train", output_dir=predict_dir, model_loader=_tiny_model_loader)
            self.assertTrue((predict_dir / "predictions.partial.jsonl").is_file())
            self.assertFalse((predict_dir / "predictions.jsonl").exists())
            run_report = json.loads((predict_dir / "run.json").read_text("utf-8"))
            self.assertEqual(run_report["status"], "failed")
            self.assertEqual(run_report["completed_records"], 1)

    def test_long_reference_is_never_silently_truncated(self):
        processor = _real_processor()
        decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        long_text = "αβγδεζηθικλμνξοπρστυφχψω " * 60  # far longer than a real HParl utterance
        labels = whisper_finetune.build_labels(processor, [long_text], language="el", task="transcribe",
                                                decoder_start_token_id=decoder_start_token_id)
        tokenizer = processor.tokenizer
        full_ids = tokenizer(long_text, padding=False, truncation=False)["input_ids"]
        # Every real (non -100) label id must come from the untruncated encoding
        # (minus the leading start token, which build_labels strips -- see above).
        kept = [token for token in labels[0].tolist() if token != -100]
        self.assertEqual(kept, full_ids[1:])


if __name__ == "__main__":
    unittest.main()
