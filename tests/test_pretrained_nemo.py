"""NeMo boundary tests without network access, model downloads or GPU use."""

from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from greek_asr.pretrained.nemo import load_backend


class NemoAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.audio = Path(self.tmp.name) / "audio.wav"
        self.audio.write_bytes(b"fixture: adapter does not decode audio")
        self.device = SimpleNamespace(type="cuda")
        self.torch = SimpleNamespace(
            device=Mock(return_value=self.device),
            cuda=SimpleNamespace(is_available=Mock(return_value=True)),
            inference_mode=nullcontext,
        )
        self.download = Mock(return_value="/cache/pinned/model.nemo")
        self.model = Mock()
        self.model_class = Mock()
        self.model_class.restore_from.return_value = self.model
        self.runtime = patch(
            "greek_asr.pretrained.nemo._import_runtime",
            return_value=(self.torch, self.download, self.model_class),
        )
        self.runtime.start()
        self.addCleanup(self.runtime.stop)
        self.revision = "d455706339a6b32e1aa40f82c713a482a0c938e2"

    def load(self, model_id="nvidia/canary-1b-v2", **kwargs):
        return load_backend(
            model_id=model_id, revision=self.revision, device="cuda:0", **kwargs
        )

    def test_pinned_archive_restored_once_and_greek_transcription_selected(self):
        backend = self.load(model_dir=Path("model-cache"))
        self.model.transcribe.return_value = [SimpleNamespace(text="Καλημέρα.")]
        self.assertEqual(backend.transcribe(self.audio), "Καλημέρα.")
        self.assertEqual(backend.transcribe(self.audio), "Καλημέρα.")
        self.download.assert_called_once_with(
            repo_id="nvidia/canary-1b-v2",
            filename="canary-1b-v2.nemo",
            revision=self.revision,
            cache_dir="model-cache",
        )
        self.model_class.restore_from.assert_called_once_with(
            restore_path="/cache/pinned/model.nemo", map_location=self.device
        )
        self.model.eval.assert_called_once_with()
        self.model.transcribe.assert_called_with(
            [str(self.audio.resolve())], batch_size=1, source_lang="el", target_lang="el"
        )

    def test_parakeet_retains_automatic_language_and_plain_text_result(self):
        backend = self.load("nvidia/parakeet-tdt-0.6b-v3")
        self.model.transcribe.return_value = ["Ένα δύο"]
        self.assertEqual(backend.transcribe(self.audio), "Ένα δύο")
        self.model.transcribe.assert_called_once_with(
            [str(self.audio.resolve())], batch_size=1
        )
        self.assertEqual(self.download.call_args.kwargs["filename"], "parakeet-tdt-0.6b-v3.nemo")

    def test_empty_hypothesis_is_retained_but_malformed_or_missing_is_an_error(self):
        backend = self.load()
        self.model.transcribe.return_value = [SimpleNamespace(text="")]
        self.assertEqual(backend.transcribe(self.audio), "")
        for result in ([], ["one", "two"], [SimpleNamespace(text=None)], None, "bare text"):
            with self.subTest(result=result):
                self.model.transcribe.return_value = result
                with self.assertRaises(RuntimeError):
                    backend.transcribe(self.audio)

    def test_source_file_required_and_runtime_failures_not_hidden(self):
        backend = self.load()
        with self.assertRaises(FileNotFoundError):
            backend.transcribe(self.audio.with_name("missing.wav"))
        self.model.transcribe.assert_not_called()
        self.model.transcribe.side_effect = RuntimeError("GPU exhausted")
        with self.assertRaisesRegex(RuntimeError, "GPU exhausted"):
            backend.transcribe(self.audio)

    def test_no_cpu_fallback_when_gpu_requested(self):
        self.torch.cuda.is_available.return_value = False
        with self.assertRaisesRegex(RuntimeError, "refusing CPU fallback"):
            self.load()
        self.download.assert_not_called()

    def test_unpinned_or_wrong_family_model_rejected_before_download(self):
        for model_id, revision in (
            ("nvidia/canary-1b-v2", "main"),
            ("some/bare-encoder", self.revision),
        ):
            with self.subTest(model_id=model_id, revision=revision):
                with self.assertRaises(ValueError):
                    load_backend(model_id=model_id, revision=revision, device="cpu")
        self.download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
