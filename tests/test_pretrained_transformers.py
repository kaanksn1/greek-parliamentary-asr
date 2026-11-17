"""Adapter contracts without downloading weights or requiring a GPU."""
from __future__ import annotations

import contextlib
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

from greek_asr.pretrained import qwen, whisper

SHA = "a" * 40


def dependencies():
    torch = types.SimpleNamespace(float32="float32", inference_mode=contextlib.nullcontext,
                                  cuda=types.SimpleNamespace(is_available=lambda: True))
    snapshot = MagicMock(return_value="pinned-snapshot")
    hub = types.SimpleNamespace(snapshot_download=snapshot)
    return torch, snapshot, hub


class WhisperTests(unittest.TestCase):
    def test_snapshot_is_pinned_and_loaded_once_for_repeated_audio(self):
        torch, snapshot, hub = dependencies()
        processor = MagicMock()
        processor.return_value = {"input_features": MagicMock(), "attention_mask": MagicMock()}
        processor.batch_decode.return_value = ["δοκιμή"]
        model = MagicMock()
        model.to.return_value = model
        model.eval.return_value = model
        classes = types.SimpleNamespace(
            WhisperProcessor=types.SimpleNamespace(from_pretrained=MagicMock(return_value=processor)),
            WhisperForConditionalGeneration=types.SimpleNamespace(from_pretrained=MagicMock(return_value=model)),
        )
        np = MagicMock()
        with patch.dict("sys.modules", {"torch": torch, "huggingface_hub": hub,
                                        "transformers": classes, "numpy": np}):
            backend = whisper.load_backend(model_id="openai/whisper-small", revision=SHA,
                                           device="cuda", model_dir=Path("cache"))
            with patch.object(whisper, "_read_pcm16", return_value=(b"\0\0", 16000)):
                self.assertEqual(backend.transcribe(Path("first.wav")), "δοκιμή")
                self.assertEqual(backend.transcribe(Path("second.wav")), "δοκιμή")
        self.assertEqual(snapshot.call_args.kwargs["revision"], SHA)
        self.assertFalse(snapshot.call_args.kwargs["token"])
        self.assertEqual(snapshot.call_count, 1)
        self.assertEqual(classes.WhisperForConditionalGeneration.from_pretrained.call_count, 1)
        self.assertTrue(classes.WhisperForConditionalGeneration.from_pretrained.call_args.kwargs["use_safetensors"])
        kwargs = model.generate.call_args.kwargs
        self.assertEqual(kwargs["language"], "el")
        self.assertEqual(kwargs["task"], "transcribe")
        self.assertFalse(kwargs["do_sample"])
        self.assertNotIn("labels", kwargs)
        self.assertNotIn("prompt_ids", kwargs)
        self.assertFalse(processor.call_args.kwargs["truncation"])
        self.assertEqual(processor.call_args.kwargs["padding"], "max_length")

    def test_long_audio_is_not_silently_cut_to_first_thirty_seconds(self):
        processor = MagicMock()
        processor.return_value = {"input_features": MagicMock()}
        processor.batch_decode.return_value = [""]
        model = MagicMock()
        torch, _, _ = dependencies()
        with patch.dict("sys.modules", {"torch": torch, "numpy": MagicMock()}), \
                patch.object(whisper, "_read_pcm16", return_value=(b"\0\0", 31*16000)):
            self.assertEqual(whisper.WhisperBackend(processor, model, "cpu").transcribe(Path("long.wav")), "")
        self.assertFalse(processor.call_args.kwargs["truncation"])
        self.assertEqual(processor.call_args.kwargs["padding"], "longest")
        self.assertTrue(model.generate.call_args.kwargs["return_timestamps"])

    def test_source_wav_is_read_unchanged_and_wrong_format_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"audio.wav"
            with wave.open(str(path), "wb") as handle:
                handle.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                handle.writeframes(b"\x01\x00\xff\xff")
            before = path.read_bytes()
            self.assertEqual(whisper._read_pcm16(path), (b"\x01\x00\xff\xff", 2))
            self.assertEqual(path.read_bytes(), before)
            with wave.open(str(path), "wb") as handle:
                handle.setparams((1, 2, 22050, 0, "NONE", "not compressed"))
                handle.writeframes(b"\0\0")
            with self.assertRaisesRegex(ValueError, "16 kHz"):
                whisper._read_pcm16(path)


class QwenTests(unittest.TestCase):
    def test_pinned_greek_audio_only_load_and_transcription(self):
        torch, snapshot, hub = dependencies()
        model = MagicMock()
        model.transcribe.return_value = [types.SimpleNamespace(text="καλημέρα", language="Greek")]
        factory = MagicMock(return_value=model)
        package = types.SimpleNamespace(Qwen3ASRModel=types.SimpleNamespace(from_pretrained=factory))
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory)/"original.wav"
            audio.write_bytes(b"audio")
            with patch.dict("sys.modules", {"torch": torch, "huggingface_hub": hub, "qwen_asr": package}):
                backend = qwen.load_backend(model_id="Qwen/Qwen3-ASR-0.6B", revision=SHA, device="cuda:0")
                self.assertEqual(backend.transcribe(audio), "καλημέρα")
        self.assertEqual(snapshot.call_args.kwargs["revision"], SHA)
        self.assertTrue(factory.call_args.kwargs["local_files_only"])
        self.assertTrue(factory.call_args.kwargs["use_safetensors"])
        self.assertEqual(factory.call_args.kwargs["attn_implementation"], "eager")
        self.assertEqual(model.transcribe.call_args.kwargs,
                         {"audio": str(audio), "language": "Greek", "context": "", "return_time_stamps": False})
        self.assertFalse(model.model.generation_config.do_sample)
        self.assertEqual(model.model.generation_config.num_beams, 1)

    def test_empty_text_is_retained_but_missing_or_multiple_results_are_errors(self):
        torch, _, _ = dependencies()
        model = MagicMock()
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory)/"original.wav"
            audio.touch()
            with patch.dict("sys.modules", {"torch": torch}):
                backend = qwen.QwenBackend(model)
                model.transcribe.return_value = [types.SimpleNamespace(text="")]
                self.assertEqual(backend.transcribe(audio), "")
                for results in ([], [types.SimpleNamespace(text=None)],
                                [types.SimpleNamespace(text="a"), types.SimpleNamespace(text="b")]):
                    model.transcribe.return_value = results
                    with self.assertRaisesRegex(ValueError, "single-record"):
                        backend.transcribe(audio)

    def test_cuda_request_has_no_silent_cpu_fallback(self):
        torch, _, hub = dependencies()
        torch.cuda.is_available = lambda: False
        fake_classes = types.SimpleNamespace(WhisperProcessor=MagicMock(), WhisperForConditionalGeneration=MagicMock())
        with patch.dict("sys.modules", {"torch": torch, "huggingface_hub": hub,
                                        "qwen_asr": types.SimpleNamespace(Qwen3ASRModel=MagicMock()),
                                        "transformers": fake_classes}):
            for module in (qwen, whisper):
                with self.subTest(backend=module.__name__), self.assertRaisesRegex(RuntimeError, "no CPU fallback"):
                    module.load_backend(model_id="example", revision=SHA, device="cuda")

    def test_mutable_revision_rejected_before_imports_or_download(self):
        for module in (qwen, whisper):
            with self.subTest(backend=module.__name__), self.assertRaisesRegex(ValueError, "immutable"):
                module.load_backend(model_id="example", revision="main", device="cpu")


if __name__ == "__main__":
    unittest.main()
