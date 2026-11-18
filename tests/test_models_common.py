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


if __name__ == "__main__":
    unittest.main()
