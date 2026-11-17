"""Verify complete-cohort inference and raw-audio contracts without ASR packages."""
from __future__ import annotations

import copy
import json
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest.mock import Mock, patch

from greek_asr import infer


class InferenceContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "project"
        self.root.mkdir()
        self.audio = self.root / "data/audio/a.wav"
        self.audio.parent.mkdir(parents=True)
        with wave.open(str(self.audio), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes(b"\0\0" * 160)
        self.manifest = self.root / "data/manifests/train.jsonl"
        self.manifest.parent.mkdir()
        self.record = {
            "id": "hparl:fixture-a", "source": "hparl", "original_id": "fixture-a",
            "source_split": "train", "split": "train",
            "audio_path": self.audio.relative_to(self.root).as_posix(),
            "audio_sha256": infer.sha256(self.audio),
            "sampling_rate": 16000, "channels": 1, "num_frames": 160,
            "duration_seconds": 0.01, "source_recording_id": None, "speaker_id": None,
        }
        self.write_manifest([self.record])
        source_lock = json.loads((infer.ROOT / "configs/model_sources.lock.json").read_text(encoding="utf-8-sig"))
        self.model = next(row for row in source_lock["models"] if row["model_id"] == "openai/whisper-small")
        (self.root / "configs").mkdir()
        (self.root / "configs/model_sources.lock.json").write_text(
            json.dumps({"models": [self.model]}), encoding="utf-8")
        # No torch/numpy/model installation is needed to test the shared runner.
        fake_torch = types.ModuleType("torch")
        fake_torch.manual_seed = Mock()
        fake_torch.set_num_threads = Mock()
        fake_torch.cuda = types.SimpleNamespace(is_available=Mock(return_value=False))
        fake_torch.version = types.SimpleNamespace(hip=None)
        self.torch = fake_torch
        torch_patch = patch.dict("sys.modules", {"torch": fake_torch})
        torch_patch.start()
        self.addCleanup(torch_patch.stop)
        environment_patch = patch.object(infer, "_environment", return_value={"fixture_environment": True})
        environment_patch.start()
        self.addCleanup(environment_patch.stop)

    def write_manifest(self, rows):
        self.manifest.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    def run_fixture(self, backend, *, output="result", **kwargs):
        return infer.run(manifest=self.manifest, backend_name="whisper", split="train",
                         output=self.root / output, project_root=self.root, device="cpu",
                         loader=Mock(return_value=backend), **kwargs)

    def second_record(self):
        row = copy.deepcopy(self.record)
        row.update(id="hparl:fixture-b", original_id="fixture-b")
        return row

    def test_namespaces_allow_same_original_id_but_reject_duplicate_full_id(self):
        css = copy.deepcopy(self.record)
        css.update(id="css10_el:fixture-a", source="css10_el", source_split="unsplit")
        self.write_manifest([self.record, css])
        rows = infer.load_manifest(self.manifest, self.root, "train")
        self.assertEqual([row["id"] for row in rows], ["hparl:fixture-a", "css10_el:fixture-a"])
        self.write_manifest([self.record, copy.deepcopy(self.record)])
        with self.assertRaisesRegex(ValueError, "duplicate ID"):
            infer.load_manifest(self.manifest, self.root, "train")

    def test_id_must_match_actual_source_and_original_id(self):
        for bad_id in ("fixture-a", "css10_el:fixture-a", "hparl:another-id"):
            with self.subTest(identifier=bad_id):
                self.write_manifest([{**self.record, "id": bad_id}])
                with self.assertRaisesRegex(ValueError, "ID must be source:original_id"):
                    infer.load_manifest(self.manifest, self.root, "train")

    def test_modified_audio_is_rejected_before_loading_model(self):
        contents = bytearray(self.audio.read_bytes())
        contents[-1] ^= 1
        self.audio.write_bytes(contents)
        loader = Mock()
        with self.assertRaisesRegex(ValueError, "audio hash mismatch"):
            infer.run(manifest=self.manifest, backend_name="whisper", split="train",
                      output=self.root / "result", project_root=self.root, device="cpu", loader=loader)
        loader.assert_not_called()
        self.assertFalse((self.root / "result").exists())

    def test_external_or_absolute_audio_path_is_rejected(self):
        outside = self.root.parent / "outside.wav"
        outside.write_bytes(self.audio.read_bytes())
        for path in ("../outside.wav", str(outside.resolve()), "C:/outside.wav"):
            with self.subTest(path=path):
                self.write_manifest([{**self.record, "audio_path": path}])
                with self.assertRaisesRegex(ValueError, "project-relative|escapes project"):
                    infer.load_manifest(self.manifest, self.root, "train")

    def test_css10_cannot_enter_development_or_test(self):
        for split in ("dev", "test"):
            with self.subTest(split=split):
                row = {**self.record, "id": "css10_el:fixture-a", "source": "css10_el",
                       "source_split": "unsplit", "split": split}
                self.write_manifest([row])
                with self.assertRaisesRegex(ValueError, "CSS10 is training-only"):
                    infer.load_manifest(self.manifest, self.root, split)

    def test_hparl_official_split_is_preserved(self):
        self.write_manifest([{**self.record, "split": "dev", "source_split": "train"}])
        with self.assertRaisesRegex(ValueError, "official split must be preserved"):
            infer.load_manifest(self.manifest, self.root, "dev")

    def test_invalid_duration_and_frame_metadata_are_rejected(self):
        for updates in ({"duration_seconds": 0}, {"duration_seconds": -1},
                        {"duration_seconds": float("nan")}, {"duration_seconds": float("inf")},
                        {"duration_seconds": 0.02}, {"num_frames": True}, {"sampling_rate": 0}):
            with self.subTest(updates=updates):
                self.write_manifest([{**self.record, **updates}])
                with self.assertRaisesRegex(ValueError, "invalid"):
                    infer.load_manifest(self.manifest, self.root, "train")

    def test_manifest_metadata_must_match_actual_pcm_wav_header(self):
        # Internally consistent metadata is still wrong if it misdescribes source audio.
        for updates in ({"sampling_rate": 8000, "duration_seconds": 0.02},
                        {"channels": 2}, {"num_frames": 320, "duration_seconds": 0.02}):
            with self.subTest(updates=updates):
                self.write_manifest([{**self.record, **updates}])
                with self.assertRaisesRegex(ValueError, "audio|WAV|header|metadata"):
                    infer.load_manifest(self.manifest, self.root, "train")

    def test_every_id_has_output_including_empty_text_and_no_gold_reaches_backend(self):
        first = {**self.record, "reference": "GOLD MUST NEVER BE MODEL INPUT", "annotation": "gold"}
        second = self.second_record()
        self.write_manifest([first, second])
        backend = types.SimpleNamespace(settings={"fixture": True},
                                        transcribe=Mock(side_effect=["", "Greek model output"]))
        loader = Mock(return_value=backend)
        report = infer.run(manifest=self.manifest, backend_name="whisper", split="train",
                           output=self.root / "result", project_root=self.root, device="cpu", loader=loader)
        predictions = [json.loads(line) for line in (self.root / "result/predictions.jsonl").read_text(
            encoding="utf-8").splitlines()]
        self.assertEqual(predictions, [{"id": first["id"], "text": ""},
                                       {"id": second["id"], "text": "Greek model output"}])
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["completed_records"], 2)
        self.assertFalse(report["references_used_as_model_input"])
        self.assertIsNone(report["wer_cer"])
        self.assertFalse((self.root / "result/predictions.partial.jsonl").exists())
        self.assertEqual(loader.call_args.kwargs["revision"], self.model["revision"])
        for call in backend.transcribe.call_args_list:
            self.assertEqual(call.args, (self.audio.resolve(),))
            self.assertEqual(call.kwargs, {})
        self.assertNotIn("reference", infer.load_manifest(self.manifest, self.root, "train")[0])

    def test_backend_failure_leaves_failed_partial_run_not_successful_predictions(self):
        self.write_manifest([self.record, self.second_record()])
        backend = types.SimpleNamespace(transcribe=Mock(side_effect=["first result", RuntimeError("fixture failure")]))
        with self.assertRaisesRegex(RuntimeError, "fixture failure"):
            self.run_fixture(backend)
        output = self.root / "result"
        self.assertFalse((output / "predictions.jsonl").exists())
        partial = [json.loads(line) for line in (output / "predictions.partial.jsonl").read_text(
            encoding="utf-8").splitlines()]
        self.assertEqual(partial, [{"id": self.record["id"], "text": "first result"}])
        report = json.loads((output / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["expected_records"], 2)
        self.assertEqual(report["completed_records"], 1)
        self.assertEqual(report["error_type"], "RuntimeError")

    def test_nonstring_prediction_is_failure_instead_of_record_drop(self):
        with self.assertRaisesRegex(ValueError, "prediction must be a string"):
            self.run_fixture(types.SimpleNamespace(transcribe=Mock(return_value=None)))
        self.assertFalse((self.root / "result/predictions.jsonl").exists())
        report = json.loads((self.root / "result/run.json").read_text(encoding="utf-8"))
        self.assertEqual((report["status"], report["completed_records"]), ("failed", 0))

    def test_audio_changed_during_inference_cannot_be_marked_complete(self):
        def mutate_audio(path):
            data = bytearray(path.read_bytes())
            data[-1] ^= 1
            path.write_bytes(data)
            return "prediction"
        with self.assertRaisesRegex(ValueError, "audio changed during inference"):
            self.run_fixture(types.SimpleNamespace(transcribe=Mock(side_effect=mutate_audio)))
        self.assertFalse((self.root / "result/predictions.jsonl").exists())
        report = json.loads((self.root / "result/run.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "failed")

    def test_manifest_changed_during_inference_cannot_be_marked_complete(self):
        def mutate_manifest(_path):
            self.manifest.write_text(self.manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            return "prediction"
        with self.assertRaisesRegex(ValueError, "manifest changed during inference"):
            self.run_fixture(types.SimpleNamespace(transcribe=Mock(side_effect=mutate_manifest)))
        self.assertFalse((self.root / "result/predictions.jsonl").exists())

    def test_existing_output_is_preserved_without_loading_model(self):
        output = self.root / "result"
        output.mkdir()
        marker = output / "old-run.txt"
        marker.write_text("existing run", encoding="utf-8")
        loader = Mock()
        with self.assertRaises(FileExistsError):
            infer.run(manifest=self.manifest, backend_name="whisper", split="train", output=output,
                      project_root=self.root, device="cpu", loader=loader)
        loader.assert_not_called()
        self.assertEqual(marker.read_text(encoding="utf-8"), "existing run")

    def test_test_guards_prevent_model_loading_and_output_creation(self):
        loader = Mock()
        for kwargs, message in (({}, "requires --allow-test"),
                                ({"allow_test": True}, "requires --test-protocol")):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                infer.run(manifest=self.root / "missing-test-manifest.jsonl", backend_name="whisper",
                          split="test", output=self.root / "result", project_root=self.root,
                          device="cpu", loader=loader, **kwargs)
        loader.assert_not_called()
        self.assertFalse((self.root / "result").exists())


if __name__ == "__main__":
    unittest.main()
