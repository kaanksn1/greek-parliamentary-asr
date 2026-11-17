"""Check real pilot lineage and reject independently corrupted copies."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from greek_asr.pilot_data import audit_pilot

PILOT = Path(__file__).resolve().parents[1] / "data/raw/pilot"


class PilotDataTests(unittest.TestCase):
    def test_real_three_record_pilot_round_trip(self):
        records, report = audit_pilot(PILOT)
        self.assertEqual(len(records), 3)
        self.assertEqual(report["audio_files_checked"], 3)
        self.assertAlmostEqual(report["total_audio_seconds"], 6.61)
        self.assertEqual(report["source_split"], "train")
        self.assertEqual(set(report["inputs_sha256"]),
                         {"manifest.jsonl", "source_summary.json", "selected_source_lines.json"})

    def test_corrupted_audio_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            copy = Path(temp) / "pilot"
            shutil.copytree(PILOT, copy)
            audio = copy / "audio/train-pilot-0001.wav"
            corrupted = bytearray(audio.read_bytes())
            corrupted[-1] ^= 1
            audio.write_bytes(corrupted)
            with self.assertRaisesRegex(ValueError, "audio hash"):
                audit_pilot(copy)

    def test_reference_lineage_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            copy = Path(temp) / "pilot"
            shutil.copytree(PILOT, copy)
            path = copy / "manifest.jsonl"
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            rows[0]["reference"] += " changed"
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source lineage mismatch"):
                audit_pilot(copy)

    def test_annotation_marker_is_flagged_without_editing_reference(self):
        before = (PILOT / "manifest.jsonl").read_bytes()
        records, report = audit_pilot(PILOT)
        marker_row = next(row for row in records if "<spoken_noise>" in row["reference"])
        self.assertEqual(report["reference_annotation_tags"][marker_row["id"]], ["<spoken_noise>"])
        self.assertEqual(report["quality_status"], "requires_reference_review")
        self.assertEqual((PILOT / "manifest.jsonl").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
