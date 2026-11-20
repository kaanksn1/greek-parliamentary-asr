"""Tests for greek_asr.models.common: real-audio join, vocab, padding, subset
selection, split stats and prediction I/O. Uses the three real train-pilot
clips already committed under data/raw/pilot/ as the only real audio fixture;
these are a synthetic test manifest, not a claim about real dev/test data.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from greek_asr.models import common

try:
    import torch  # noqa: F401
    import torchaudio  # noqa: F401
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
     "num_frames": 40160, "duration_seconds": 2.51, "reference": "εμείς μπορούμε σήμερα",
     "speaker_id": "alekshs_tsipras"},
    {"id": "hparl:dhmhtrios_gakhs-olomeleia_15_01512500_01515500_3_1",
     "original_id": "dhmhtrios_gakhs-olomeleia_15_01512500_01515500_3_1",
     "audio_path": "data/raw/pilot/audio/train-pilot-0002.wav",
     "audio_sha256": "558c1a270332a49d97d3e8993a307499fd7b719c05e6d9d6493ce2dde77896b5",
     "num_frames": 35200, "duration_seconds": 2.2, "reference": "σε αυτήν την οικονομική δράση",
     "speaker_id": "dhmhtrios_gakhs"},
    {"id": "hparl:iwannhs_balafas-olomeleia_47_02640000_02643000_1_2",
     "original_id": "iwannhs_balafas-olomeleia_47_02640000_02643000_1_2",
     "audio_path": "data/raw/pilot/audio/train-pilot-0003.wav",
     "audio_sha256": "719b0a4fb9f5205d8cab4fc74904fe796da07ebe30ae242ccefd76955982dbe6",
     "num_frames": 30400, "duration_seconds": 1.9, "reference": "τους ξεφεύγει η αντικατάσταση της <spoken_noise>",
     "speaker_id": "iwannhs_balafas"},
]


def _write_manifest_and_references(directory: Path, rows, split: str) -> tuple[Path, Path]:
    manifest_path, references_path = directory / "manifest.jsonl", directory / "references.jsonl"
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


class LoadUtterancesTests(unittest.TestCase):
    def test_real_pilot_audio_joins_with_references_and_rejects_test_split(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path, references_path = _write_manifest_and_references(Path(directory), PILOT, "train")
            utterances = common.load_utterances(manifest_path, references_path, project_root=ROOT, split="train")
        self.assertEqual(len(utterances), 3)
        by_id = {u.id: u for u in utterances}
        self.assertEqual(by_id["hparl:alekshs_tsipras-olomeleia_37_00240000_00243000_4_1"].reference,
                         "εμείς μπορούμε σήμερα")
        self.assertTrue(all(u.audio_path.is_file() for u in utterances))
        with self.assertRaisesRegex(ValueError, "frozen test split"):
            common.load_utterances(manifest_path, references_path, project_root=ROOT, split="test")

    def test_missing_reference_id_is_never_silently_dropped(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path, references_path = _write_manifest_and_references(Path(directory), PILOT, "train")
            # Drop one reference row entirely instead of leaving it blank.
            kept = [row for row in PILOT if row["id"] != PILOT[0]["id"]]
            with references_path.open("w", encoding="utf-8") as stream:
                for row in kept:
                    stream.write(json.dumps({"id": row["id"], "reference": row["reference"]}, ensure_ascii=False) + "\n")
            with self.assertRaisesRegex(ValueError, "missing="):
                common.load_utterances(manifest_path, references_path, project_root=ROOT, split="train")


@unittest.skipUnless(_TORCH_AVAILABLE, _SKIP_REASON)
class WaveformTests(unittest.TestCase):
    def test_real_wav_decodes_and_resamples_deterministically_without_editing_source(self):
        path = ROOT / "data/raw/pilot/audio/train-pilot-0001.wav"
        before = path.read_bytes()
        waveform = common.load_waveform(path, target_sample_rate=16000)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(waveform.dim(), 1)
        self.assertEqual(waveform.shape[0], 40160)
        resampled = common.load_waveform(path, target_sample_rate=8000)
        self.assertLess(resampled.shape[0], waveform.shape[0])
        self.assertEqual(path.read_bytes(), before)

    @staticmethod
    def _write_wav(path: Path, *, sample_width: int, sample_rate: int, samples: list[int]) -> None:
        import struct
        import wave

        codes = {2: "h", 4: "i"}
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(sample_width)
            handle.setframerate(sample_rate)
            handle.writeframes(struct.pack(f"<{len(samples)}{codes[sample_width]}", *samples))

    def test_pcm32_is_decoded_at_its_real_32_bit_width_not_misread_as_int16(self):
        """Regression for the CSS10 bug: a 32-bit PCM WAV was previously decoded
        as int16, doubling the apparent frame count and corrupting the amplitude
        (docs/reviews/ADNAN_0437631_GPU.json "P1 - CSS10 PCM32 yanlış çözülüyor").
        """
        import torch

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pcm32.wav"
            samples = [0, 2 ** 30, -(2 ** 30), 2 ** 31 - 1]
            self._write_wav(path, sample_width=4, sample_rate=16000, samples=samples)
            before = path.read_bytes()
            waveform = common.load_waveform(path, target_sample_rate=16000)
            self.assertEqual(path.read_bytes(), before)  # source bytes never rewritten
        self.assertEqual(waveform.numel(), len(samples))  # not 2x from a wrong 16-bit read
        expected = torch.tensor([x / 2 ** 31 for x in samples], dtype=torch.float32)
        self.assertTrue(torch.allclose(waveform, expected, atol=1e-6))

    def test_pcm32_resamples_in_memory_without_touching_the_source_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pcm32.wav"
            self._write_wav(path, sample_width=4, sample_rate=16000,
                            samples=[0, 2 ** 29, -(2 ** 29), 2 ** 28, -(2 ** 28), 100, -100, 2 ** 30])
            before = path.read_bytes()
            full_rate = common.load_waveform(path, target_sample_rate=16000)
            resampled = common.load_waveform(path, target_sample_rate=8000)
            self.assertEqual(path.read_bytes(), before)
        self.assertLess(resampled.shape[0], full_rate.shape[0])

    def test_unsupported_sample_width_raises_a_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pcm24.wav"
            # 24-bit PCM: stdlib `wave` will happily write this width, but
            # load_waveform must refuse to guess a scale for it.
            with __import__("wave").open(str(path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(3)
                handle.setframerate(16000)
                handle.writeframes(b"\x00\x00\x00" * 4)
            with self.assertRaisesRegex(ValueError, "unsupported PCM sample width"):
                common.load_waveform(path, target_sample_rate=16000)

    def test_ieee_float32_decodes_real_amplitudes_instead_of_integer_bits(self):
        import struct
        import wave

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "float.wav"
            # Hand-build a minimal WAVE_FORMAT_IEEE_FLOAT (format tag 3) header,
            # since the stdlib `wave` module cannot write this format itself.
            data = struct.pack("<4f", 0.0, 0.5, -0.5, 1.0)
            fmt_chunk = struct.pack("<HHIIHH", 3, 1, 16000, 16000 * 4, 4, 32)
            with path.open("wb") as stream:
                stream.write(b"RIFF")
                stream.write(struct.pack("<I", 4 + 8 + len(fmt_chunk) + 8 + len(data)))
                stream.write(b"WAVE")
                stream.write(b"fmt ")
                stream.write(struct.pack("<I", len(fmt_chunk)))
                stream.write(fmt_chunk)
                stream.write(b"data")
                stream.write(struct.pack("<I", len(data)))
                stream.write(data)
            before = path.read_bytes()
            decoded = common.load_waveform(path, target_sample_rate=16000)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(decoded.tolist(), [0.0, 0.5, -0.5, 1.0])


class VocabularyTests(unittest.TestCase):
    def test_vocab_is_learned_only_from_given_references_and_round_trips(self):
        vocab = common.build_char_vocabulary(
            ["εμείς μπορούμε", "σήμερα"], specials=[common.PAD, common.SOS, common.EOS, common.UNK])
        self.assertIn("ε", vocab.stoi)
        self.assertNotIn("ζ", vocab.stoi)  # never seen in the given references
        encoded = vocab.encode("σήμερα")
        self.assertEqual(vocab.decode(encoded), "σήμερα")
        restored = common.Vocabulary.from_json(vocab.to_json())
        self.assertEqual(restored.itos, vocab.itos)

    def test_out_of_vocabulary_character_without_unk_raises(self):
        vocab = common.build_char_vocabulary(["αβγ"], specials=[common.PAD])
        with self.assertRaisesRegex(ValueError, "out of vocabulary"):
            vocab.encode("δ")


@unittest.skipUnless(_TORCH_AVAILABLE, _SKIP_REASON)
class PaddingTests(unittest.TestCase):
    def test_pad_sequences_preserves_real_lengths_and_zero_pads(self):
        import torch

        sequences = [torch.ones(3, 2), torch.ones(5, 2), torch.ones(1, 2)]
        padded, lengths = common.pad_sequences(sequences)
        self.assertEqual(padded.shape, (3, 5, 2))
        self.assertEqual(lengths.tolist(), [3, 5, 1])
        self.assertTrue(torch.all(padded[2, 1:] == 0))
        mask = common.padding_mask(lengths, 5)
        self.assertEqual(mask[2].tolist(), [True, False, False, False, False])

    def test_padding_mask_is_built_on_the_lengths_tensor_device_not_cpu_by_default(self):
        """Regression for the GPU pilot crash: a CPU-only torch.arange compared
        against a GPU-resident lengths tensor raised "Expected all tensors to be
        on the same device" (docs/reviews/ADNAN_0437631_GPU.json finding 2). No
        real GPU is required to test the device *contract*: torch's "meta" device
        (shape-only, no data) exercises the exact same device-mismatch code path.
        """
        import torch

        meta_lengths = torch.tensor([2, 4], device="meta")
        mask = common.padding_mask(meta_lengths, 4)
        self.assertEqual(mask.device.type, "meta")
        self.assertEqual(mask.shape, (2, 4))


@unittest.skipUnless(_TORCH_AVAILABLE, _SKIP_REASON)
class DeviceSelectionTests(unittest.TestCase):
    def test_cpu_is_always_accepted(self):
        device = common.resolve_device("cpu")
        self.assertEqual(device.type, "cpu")

    def test_requesting_an_unavailable_cuda_device_raises_instead_of_falling_back(self):
        import torch

        if torch.cuda.is_available():
            raise unittest.SkipTest("this machine has a real CUDA/ROCm device; "
                                    "the no-GPU error path cannot be exercised here")
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            common.resolve_device("cuda")

    def test_move_batch_to_device_only_touches_tensor_values(self):
        import torch

        batch = {"ids": ["a", "b"], "features": torch.zeros(2, 3), "feature_lengths": torch.tensor([3, 3])}
        moved = common.move_batch_to_device(batch, torch.device("meta"))
        self.assertEqual(moved["ids"], ["a", "b"])
        self.assertEqual(moved["features"].device.type, "meta")
        self.assertEqual(moved["feature_lengths"].device.type, "meta")

    def test_checkpoint_loads_with_map_location_regardless_of_training_device(self):
        """The checkpoint itself must not pin a training-time device: saving
        a model's state and loading it back must work even when the simulated
        'training device' differs from the load-time device.
        """
        import torch
        import torch.nn as nn

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            model = nn.Linear(4, 2)
            vocab = common.build_char_vocabulary(["αβ"], specials=[common.PAD])
            common.save_checkpoint(path, model=model, optimizer=None, scheduler=None,
                                   step=1, config={"device": "meta"}, vocab=vocab)
            checkpoint = common.load_checkpoint(path)
        self.assertTrue(all(tensor.device.type == "cpu" for tensor in checkpoint["model_state"].values()))


class PredictionWriterTests(unittest.TestCase):
    def test_genuine_empty_prediction_is_kept_and_missing_id_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "predictions.jsonl"
            common.write_predictions(output, ["a", "b"], {"a": "", "b": "γεια"})
            lines = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(lines, [{"id": "a", "text": ""}, {"id": "b", "text": "γεια"}])
            with self.assertRaisesRegex(ValueError, "missing="):
                common.write_predictions(output, ["a", "b", "c"], {"a": "", "b": "γεια"})


class SubsetSelectionTests(unittest.TestCase):
    def _rows(self):
        rows = []
        for speaker in ("s1", "s2", "s3"):
            for clip in range(2):
                rows.append({"original_id": f"{speaker}-{clip}", "speaker_id": speaker, "duration_seconds": 3600.0})
        rows.append({"original_id": "no-speaker-0", "speaker_id": None, "duration_seconds": 3600.0})
        return rows

    def test_speaker_groups_are_never_split_across_the_selection_boundary(self):
        rows = self._rows()
        selected, diagnostics = common.select_grouped_subset(
            rows, group_key_priority=("speaker_id",), seed=42, target_count=2)
        selected_speakers = {row["speaker_id"] for row in selected}
        for speaker in selected_speakers:
            full_group = [row for row in rows if row["speaker_id"] == speaker]
            self.assertTrue(all(row in selected for row in full_group))
        self.assertEqual(diagnostics["groups_without_speaker_or_recording_id"], 1)

    def test_selection_is_deterministic_for_a_fixed_seed(self):
        rows = self._rows()
        first, _ = common.select_grouped_subset(rows, group_key_priority=("speaker_id",), seed=7, target_count=4)
        second, _ = common.select_grouped_subset(rows, group_key_priority=("speaker_id",), seed=7, target_count=4)
        self.assertEqual([row["original_id"] for row in first], [row["original_id"] for row in second])

    def test_exactly_one_target_must_be_given(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            common.select_grouped_subset(self._rows(), group_key_priority=("speaker_id",), seed=1)


class SplitStatisticsTests(unittest.TestCase):
    def test_speaker_utterance_hour_counts_and_literal_overlap(self):
        rows = [{"speaker_id": "a", "duration_seconds": 3600.0},
                {"speaker_id": None, "duration_seconds": 1800.0}]
        stats = common.split_statistics(rows)
        self.assertEqual(stats, {"utterances": 2, "known_speakers": 1,
                                 "utterances_with_unknown_speaker": 1, "total_hours": 1.5})
        overlap = common.literal_overlap(["γεια", "γεια", "καλημέρα"], ["γεια", "αντίο", "γεια"])
        self.assertEqual(overlap, {"test_examples": 3, "literal_matches": 2, "unique_matched_texts": 1})


class ProvenanceTests(unittest.TestCase):
    def test_hash_file_is_deterministic_and_matches_hashlib(self):
        import hashlib

        path = ROOT / "data/raw/pilot/audio/train-pilot-0001.wav"
        expected = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(common.hash_file(path), expected)
        self.assertEqual(common.hash_file(path), common.hash_file(path))

    def test_git_commit_hash_returns_the_actual_checked_out_commit(self):
        import subprocess

        expected = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                  check=True, text=True).stdout.strip()
        self.assertEqual(common.git_commit_hash(ROOT), expected)

    def test_git_commit_hash_is_none_outside_a_git_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(common.git_commit_hash(Path(directory)))

    def test_manifest_source_breakdown_separates_hparl_from_css10(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "train.jsonl"
            rows = [
                {"source": "hparl", "duration_seconds": 3600.0},
                {"source": "hparl", "duration_seconds": 1800.0},
                {"source": "css10_el", "duration_seconds": 900.0},
            ]
            manifest_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            breakdown = common.manifest_source_breakdown(manifest_path)
        self.assertEqual(breakdown["hparl"], {"utterances": 2, "hours": 1.5})
        self.assertEqual(breakdown["css10_el"]["utterances"], 1)
        self.assertAlmostEqual(breakdown["css10_el"]["hours"], 0.25)


@unittest.skipUnless(_TORCH_AVAILABLE, _SKIP_REASON)
class PartialPredictionWriterTests(unittest.TestCase):
    def test_full_run_renames_partial_to_final(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            writer = common.PartialPredictionWriter(output_dir, ["a", "b"])
            writer.write("a", "")
            writer.write("b", "γεια")
            writer.finish()
            self.assertFalse((output_dir / "predictions.partial.jsonl").exists())
            lines = [json.loads(line) for line in (output_dir / "predictions.jsonl").read_text("utf-8").splitlines()]
            self.assertEqual(lines, [{"id": "a", "text": ""}, {"id": "b", "text": "γεια"}])

    def test_abandoned_run_leaves_partial_file_and_no_final_file(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            writer = common.PartialPredictionWriter(output_dir, ["a", "b"])
            writer.write("a", "only-this-one")
            writer.abandon()
            self.assertTrue((output_dir / "predictions.partial.jsonl").is_file())
            self.assertFalse((output_dir / "predictions.jsonl").exists())
            lines = [json.loads(line) for line in (output_dir / "predictions.partial.jsonl").read_text("utf-8").splitlines()]
            self.assertEqual(lines, [{"id": "a", "text": "only-this-one"}])

    def test_finishing_with_a_missing_id_raises_instead_of_silently_completing(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = common.PartialPredictionWriter(Path(directory), ["a", "b"])
            writer.write("a", "x")
            with self.assertRaisesRegex(ValueError, "missing"):
                writer.finish()


if __name__ == "__main__":
    unittest.main()
