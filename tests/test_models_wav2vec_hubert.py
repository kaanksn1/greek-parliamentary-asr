"""Tests for Kemal's published-checkpoint CTC inference (wav2vec 2.0, HuBERT) and HuBERT-CTC training.

Tiny randomly initialised Hugging Face models stand in for the real
checkpoints, so nothing is downloaded; real-checkpoint runtime evidence is
kept separately under results/runtime_smoke/. Pilot clips are a code-path
fixture only.
"""
from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path

from greek_asr import infer

try:
    import torch
    import transformers  # noqa: F401

    from greek_asr.models import common, ctc_utils, hubert, wav2vec
    _HF = True
except ImportError:
    _HF = False

from tests.test_models_attention import PILOT, _write_manifest_and_references

ROOT = Path(__file__).resolve().parents[1]
SKIP = "requires torch/torchaudio/transformers"
VOCAB = {"|": 0, "α": 1, "β": 2, "λ": 3, "[UNK]": 4, "[PAD]": 5}
TINY = dict(hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32, conv_dim=(8, 8),
            conv_kernel=(10, 3), conv_stride=(5, 2), conv_bias=False, num_conv_pos_embeddings=8,
            num_conv_pos_embedding_groups=2, feat_extract_norm="group", do_stable_layer_norm=False)


def _tokenizer(directory: Path):
    from transformers import Wav2Vec2CTCTokenizer

    path = directory / "vocab.json"
    path.write_text(json.dumps(VOCAB, ensure_ascii=False), encoding="utf-8")
    return Wav2Vec2CTCTokenizer(str(path), unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")


def _feature_extractor(return_attention_mask: bool):
    from transformers import Wav2Vec2FeatureExtractor

    return Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True,
                                    return_attention_mask=return_attention_mask)


class _ScriptedModel:
    """Returns a fixed best path, to check decoding without a trained network."""

    def __init__(self, path, vocab_size=6, pad=5):
        self.config = types.SimpleNamespace(pad_token_id=pad, vocab_size=vocab_size)
        self.path = path

    def __call__(self, **inputs):
        logits = torch.full((1, len(self.path), self.config.vocab_size), -10.0)
        for t, token in enumerate(self.path):
            logits[0, t, token] = 0.0
        return types.SimpleNamespace(logits=logits)


@unittest.skipUnless(_HF, SKIP)
class PublishedCheckpointBackendTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.tokenizer = _tokenizer(self.dir)
        self.audio = ROOT / PILOT[0]["audio_path"]

    def tearDown(self):
        self.tmp.cleanup()

    def test_double_letters_separated_by_blank_survive_decoding(self):
        # λ λ [PAD] λ | α α  ->  "λλ α" (the collapse keeps the blank-separated repeat)
        backend = wav2vec.HFCTCBackend(_feature_extractor(True), self.tokenizer, _ScriptedModel([3, 3, 5, 3, 0, 1, 1]),
                                       "cpu", model_label="scripted")
        self.assertEqual(backend.transcribe(self.audio), "λλ α")

    def test_all_blank_output_is_a_genuine_empty_string(self):
        backend = wav2vec.HFCTCBackend(_feature_extractor(True), self.tokenizer, _ScriptedModel([5, 5, 5]),
                                       "cpu", model_label="scripted")
        self.assertEqual(backend.transcribe(self.audio), "")

    def test_blank_must_be_the_tokenizer_pad(self):
        with self.assertRaises(ValueError):
            wav2vec.HFCTCBackend(_feature_extractor(True), self.tokenizer, _ScriptedModel([1], pad=4), "cpu",
                                 model_label="scripted")

    def test_tiny_real_models_transcribe_original_audio(self):
        from transformers import HubertConfig, HubertForCTC, Wav2Vec2Config, Wav2Vec2ForCTC

        torch.manual_seed(0)
        for config_class, model_class in ((Wav2Vec2Config, Wav2Vec2ForCTC), (HubertConfig, HubertForCTC)):
            model = model_class(config_class(vocab_size=6, pad_token_id=5, **TINY)).eval()
            backend = wav2vec.HFCTCBackend(_feature_extractor(False), self.tokenizer, model, "cpu", model_label="tiny")
            text = backend.transcribe(self.audio)
            self.assertIsInstance(text, str)
            self.assertEqual(backend.settings["model_class"], model_class.__name__)

    def test_loader_rejects_mutable_revision_and_missing_gpu_before_download(self):
        with self.assertRaises(ValueError):
            wav2vec.load_backend(model_id="lighteternal/wav2vec2-large-xlsr-53-greek", revision="main", device="cpu")
        with self.assertRaises(ValueError):
            hubert.load_backend(model_id="ntinouldinho/hubert_greek_6gram", revision="a" * 40, device="gpu")
        if not torch.cuda.is_available():
            with self.assertRaises(RuntimeError):
                wav2vec.load_backend(model_id="lighteternal/wav2vec2-large-xlsr-53-greek", revision="a" * 40,
                                     device="cuda")

    def test_shared_runner_uses_each_backend_lock_and_records_its_code(self):
        manifest, _ = _write_manifest_and_references(self.dir, PILOT, "train")
        backend = wav2vec.HFCTCBackend(_feature_extractor(True), self.tokenizer, _ScriptedModel([1, 5, 2]), "cpu",
                                       model_label="scripted")
        expected = {"wav2vec": ("configs/model_sources.lock.json", "greek_asr/models/wav2vec.py"),
                    "hubert": ("configs/kemal_model_sources.json", "greek_asr/models/hubert.py")}
        for name, (lock, module_file) in expected.items():
            calls = []

            def loader(**kwargs):
                calls.append(kwargs)
                return backend

            report = infer.run(manifest=manifest, backend_name=name, split="train", output=self.dir / name,
                               project_root=ROOT, device="cpu", purpose="runtime-smoke", loader=loader)
            locked = json.loads((ROOT / lock).read_text(encoding="utf-8-sig"))
            revision = next(row["revision"] for row in locked["models"] if row["model_id"] == infer.BACKENDS[name][1])
            self.assertEqual(report["status"], "complete")
            self.assertEqual(calls[0]["revision"], revision)
            self.assertIn(module_file, report["code_sha256"])
            rows = [json.loads(line) for line in (self.dir / name / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual([row["text"] for row in rows], ["αβ"] * 3)

    def test_historical_source_lock_is_unchanged(self):
        recorded = {json.loads(path.read_text(encoding="utf-8"))["source_lock_sha256"]
                    for path in (ROOT / "results/runtime_smoke").glob("owner_*/run.json")}
        self.assertEqual(recorded, {common.hash_file(ROOT / "configs/model_sources.lock.json")})
        locked = json.loads((ROOT / "configs/model_sources.lock.json").read_text(encoding="utf-8-sig"))
        self.assertNotIn("ntinouldinho/hubert_greek_6gram", {row["model_id"] for row in locked["models"]})


@unittest.skipUnless(_HF, SKIP)
class HubertCtcTrainingTests(unittest.TestCase):
    def setUp(self):
        from transformers import HubertConfig, HubertModel

        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        torch.manual_seed(0)
        encoder = self.dir / "encoder"
        HubertModel(HubertConfig(**TINY)).save_pretrained(encoder, safe_serialization=True)
        _feature_extractor(False).save_pretrained(encoder)
        clean = [dict(row, reference=ctc_utils.strip_annotation_tags(row["reference"])) for row in PILOT]
        self.train_manifest, self.train_refs = _write_manifest_and_references(self.dir, clean, "train")
        self.dev_manifest, self.dev_refs = _write_manifest_and_references(self.dir, PILOT, "dev")
        self.config = {
            "encoder_local_dir": str(encoder), "train_manifest": str(self.train_manifest),
            "train_references": str(self.train_refs), "dev_manifest": str(self.dev_manifest),
            "dev_references": str(self.dev_refs), "seed": 5, "learning_rate": 0.001, "warmup_steps": 1,
            "batch_size": 2, "max_steps": 2, "dev_eval_every": 1, "mask_time_prob": 0.0,
        }

    def tearDown(self):
        self.tmp.cleanup()

    def test_output_lengths_match_transformers(self):
        from transformers import HubertConfig, HubertForCTC

        model = HubertForCTC(HubertConfig(vocab_size=6, pad_token_id=5, **TINY))
        lengths = torch.tensor([400, 401, 16000, 30400])
        self.assertTrue(torch.equal(hubert.encoder_output_lengths(model.config, lengths),
                                    model._get_feat_extract_output_lengths(lengths)))

    def test_head_training_updates_transformer_but_not_frozen_cnn_and_predicts_every_id(self):
        from transformers import HubertModel

        before = HubertModel.from_pretrained(self.dir / "encoder").state_dict()
        report = hubert.train(self.config, project_root=ROOT, output_dir=self.dir / "run")
        self.assertEqual((report["status"], report["optimizer_updates_completed"]), ("complete", 2))
        self.assertIn("own CTC fine-tuning", report["provenance"])
        model, _, vocab, metadata = hubert.load_trained(self.dir / "run/checkpoint-best", "cpu")
        after = {name.removeprefix("hubert."): value for name, value in model.state_dict().items()}
        cnn = [name for name in before if name.startswith("feature_extractor")]
        transformer = [name for name in before if name.startswith("encoder.layers")]
        self.assertTrue(cnn and all(torch.equal(before[name], after[name]) for name in cnn))
        self.assertTrue(any(not torch.equal(before[name], after[name]) for name in transformer))
        self.assertEqual(model.config.pad_token_id, vocab.stoi[common.BLANK])

        result = hubert.predict(self.dir / "run/checkpoint-best", self.dev_manifest, project_root=ROOT, split="dev",
                                output_dir=self.dir / "pred", batch_size=2)
        rows = [json.loads(line) for line in (self.dir / "pred/predictions.jsonl").read_text().splitlines()]
        self.assertEqual(result["status"], "complete")
        self.assertEqual([row["id"] for row in rows], [row["id"] for row in PILOT])

    def test_frozen_transformer_warmup_trains_only_the_head(self):
        from transformers import HubertModel

        before = HubertModel.from_pretrained(self.dir / "encoder").state_dict()
        config = dict(self.config, freeze_transformer_updates=5)  # longer than the 2-update run
        hubert.train(config, project_root=ROOT, output_dir=self.dir / "run")
        model, *_ = hubert.load_trained(self.dir / "run/checkpoint-last", "cpu")
        after = {name.removeprefix("hubert."): value for name, value in model.state_dict().items()}
        self.assertTrue(all(torch.equal(before[name], after[name]) for name in before if name in after))

    def test_missing_budget_and_test_split_are_refused(self):
        with self.assertRaises(ValueError):
            hubert.train(dict(self.config, max_steps=None), project_root=ROOT, output_dir=self.dir / "none")
        hubert.train(self.config, project_root=ROOT, output_dir=self.dir / "run")
        with self.assertRaises(ValueError):
            hubert.predict(self.dir / "run/checkpoint-best", self.dev_manifest, project_root=ROOT, split="test",
                           output_dir=self.dir / "test")
        self.assertFalse((self.dir / "test").exists())


if __name__ == "__main__":
    unittest.main()
