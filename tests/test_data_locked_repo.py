"""The committed data identity that running experiments depend on (real repository files).

Metadata checks need no audio and run everywhere. The real CSS10 audio check
runs when the original WAVs are present (after ``git lfs pull`` or a local
source extraction); on a clone without audio bytes it is skipped with the reason.
"""
from __future__ import annotations

import unittest
from pathlib import Path

from greek_asr.data import freeze, selection
from greek_asr.data.common import (LFS_POINTER, ROOT, probe_wav, read_json, read_jsonl, resolve_inside,
                                   sha256_file)

FREEZE_SHA256 = "9a680104427fa0e202229167f5eb09bf61c713ebce29c9e369413a9f230675c3"
TEST_MANIFEST_SHA256 = "b0842bfbcbffa173585adb2debf35fb359748e0bd693da5a8dfabdb69de6f65a"
DEV_MANIFEST_SHA256 = "316fbcfbfaf42a3f34a84f75427bd28ec33ee012b7a9e9a2ac10e38b8a69a29a"
DEV_REFERENCES_SHA256 = "438fdd178dd01a873a9e6adf05de8f717c22da94743527afe45c83fdcad24c4c"
LOCKED_SELECTION = {  # configs/training_data_selection.json at the owner's integration commit 45bb2b4e
    "data/manifests/subsets/train_hparl_30.5h_seed42.jsonl":
        "f9731a5679c90e55cb97902e3b2996805fb49164f9382dc09b85fa2f66ee75db",
    "data/manifests/subsets/references/train_hparl_30.5h_seed42.jsonl":
        "bcf07eb981fa21369d081e9e1b8f682e7e07fe96cbcf4fefc0794d619acc5198",
    "data/manifests/subsets/raw/train_hparl_30.5h_seed42.jsonl":
        "3991a65229c95fdf6a9f4451b6cb9888b64f341460a428850237f912d07715da",
    "data/manifests/subsets/raw/references/train_hparl_30.5h_seed42.jsonl":
        "56ac531ff4869c0209b0a06430745e33ca819d5cb961a9de6294d56d9d151539",
}
PLAN = ROOT / "configs/training_data_selection.json"
LOCKED_SELECTION_PATH = "data/manifests/subsets/train_hparl_30.5h_seed42.jsonl"


def css10_audio_present() -> bool:
    path = ROOT / "data/manifests/css10_el_train.jsonl"
    if not path.is_file():
        return False
    first = resolve_inside(ROOT, read_jsonl(path)[0]["audio_path"])
    return first.is_file() and not first.read_bytes()[:len(LFS_POINTER)] == LFS_POINTER


@unittest.skipUnless(PLAN.is_file(), "configs/training_data_selection.json is not in this checkout")
class LockedDataIdentityTests(unittest.TestCase):
    def test_locked_hparl_selection_files_are_unchanged(self):
        for relative, digest in LOCKED_SELECTION.items():
            self.assertEqual(sha256_file(ROOT / relative), digest, relative)
        chosen = selection.load_selection(ROOT)  # also checks every artifact hash in the plan
        self.assertEqual(len(chosen["session_ids"]), 18)
        self.assertEqual(len(chosen["raw_rows"]), 23615)
        clean = read_jsonl(ROOT / chosen["artifacts"]["clean_manifest"]["path"])
        self.assertEqual(len(clean), 23602)
        self.assertAlmostEqual(sum(r["duration_seconds"] for r in clean) / 3600, 30.883408, places=5)
        selection.check_raw_subset(ROOT, chosen)

    def test_dev_and_frozen_test_are_unchanged(self):
        self.assertEqual(sha256_file(ROOT / "data/manifests/dev.jsonl"), DEV_MANIFEST_SHA256)
        self.assertEqual(sha256_file(ROOT / "data/manifests/references/dev.jsonl"), DEV_REFERENCES_SHA256)
        self.assertEqual(sha256_file(ROOT / "data/manifests/test.freeze.json"), FREEZE_SHA256)
        self.assertEqual(sha256_file(ROOT / "data/manifests/test.jsonl"), TEST_MANIFEST_SHA256)
        self.assertEqual(freeze.verify_freeze(ROOT, check_audio=False)["status"], "ok")

    def test_css10_mixture_lineage_matches_committed_files(self):
        path = ROOT / selection.LINEAGE
        if not path.is_file():
            self.skipTest(f"{selection.LINEAGE} not generated yet (run `clean --selection`)")
        lineage = read_json(path)
        self.assertEqual(lineage["status"], "css10_reaudited_mixture_ready")
        self.assertEqual(lineage["audit"]["schema"], "p3-data-audit-v2")
        locked = lineage["locked_inputs_unchanged"]
        self.assertTrue(locked["selected_clean_hparl"]["reproduced_byte_identical"])
        self.assertTrue(locked["dev"]["reproduced_byte_identical"])
        current = selection.current_training_manifests(ROOT, selection.load_selection(ROOT))
        self.assertEqual(current["source"], selection.LINEAGE)
        mixed = read_jsonl(ROOT / current["mixed_manifest"]["path"])
        css10 = read_jsonl(ROOT / current["css10_manifest"]["path"])
        clean = read_jsonl(ROOT / LOCKED_SELECTION_PATH)
        self.assertEqual([r["id"] for r in mixed], sorted(r["id"] for r in [*clean, *css10]))

    @unittest.skipUnless(css10_audio_present(), "real CSS10 WAV bytes are not present (Git LFS pointers only)")
    def test_real_css10_audio_is_ieee_float32_and_matches_recorded_hashes(self):
        rows = read_jsonl(ROOT / "data/manifests/raw/css10_el.jsonl")
        self.assertEqual(len(rows), 1844)
        for row in rows:
            measured = probe_wav(resolve_inside(ROOT, row["audio_path"]))  # also rejects NaN/Inf
            self.assertEqual(measured["audio_sha256"], row["audio_sha256"], row["id"])
            self.assertEqual((measured["format_tag"], measured["sample_width_bytes"], measured["sampling_rate"],
                              measured["channels"], measured["num_frames"]),
                             (3, 4, 22050, 1, row["num_frames"]), row["id"])



if __name__ == "__main__":
    unittest.main()
