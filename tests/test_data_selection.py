"""Selected-scope audit -> clean -> CSS10 mixture -> verify on the synthetic corpus.

The fixture's CSS10 clips are IEEE float32 like the real ones. The locked
selection is written like configs/training_data_selection.json; the chain must
reproduce its clean HParl and dev byte-for-byte without any unselected train
audio, never rewrite a locked file and version a changed CSS10/mixture.
"""
from __future__ import annotations

import contextlib
import gzip
import io
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

from greek_asr.data import audit, selection
from greek_asr.data.__main__ import main as data_cli
from greek_asr.data.common import DataError, read_json, read_jsonl
from tests.data_fixtures import (CSS10, TRAIN, float_samples, make_project, run_stage_one, run_stage_two,
                                 write_selection_plan)

SELECTED = ["olomeleia_1", "olomeleia_2", "olomeleia_4", "olomeleia_8"]
LINEAGE = "configs/css10_mixture_lineage.json"


def cli(root: Path, *arguments: str) -> tuple[int, dict | None]:
    with contextlib.redirect_stdout(io.StringIO()) as output, contextlib.redirect_stderr(io.StringIO()):
        code = data_cli(["--root", str(root), *arguments])
    return code, json.loads(output.getvalue()) if code == 0 else None


def tree_bytes(root: Path, *relatives: str) -> dict[str, bytes]:
    result = {}
    for relative in relatives:
        path = root / relative
        files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
        result.update({p.relative_to(root).as_posix(): p.read_bytes() for p in files})
    return result


class SelectedScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.base = make_project(Path(cls.temporary.name) / "base")
        run_stage_one(cls.base)
        run_stage_two(cls.base)
        cls.root = cls.copy(cls.base, "selected")
        write_selection_plan(cls.root, SELECTED)
        cls.locked = tree_bytes(cls.root, "data/manifests/subsets", "data/manifests/dev.jsonl",
                                "data/manifests/references/dev.jsonl", "data/manifests/test.jsonl",
                                "data/manifests/references/test.jsonl", "data/manifests/test.freeze.json",
                                "configs/training_data_selection.json", "data/manifests/css10_el_train.jsonl")
        cls.removed = []
        for row in read_jsonl(cls.root / "data/manifests/raw/hparl_train.jsonl"):  # unused train audio absent
            if row["source_recording_id"] not in SELECTED:
                (cls.root / row["audio_path"]).unlink()
                cls.removed.append(row["audio_path"])
        cls.results = {name: cli(cls.root, name, "--selection", "configs/training_data_selection.json")
                       for name in ("audit", "clean", "verify")}

    @classmethod
    def copy(cls, source: Path, name: str) -> Path:
        target = Path(cls.temporary.name) / name
        shutil.copytree(source, target)
        return target

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def fresh(self, name: str) -> Path:
        target = Path(self.temporary.name) / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(self.root, target)
        return target

    def test_chain_runs_without_unselected_train_audio(self):
        self.assertTrue(self.removed)
        for name, (code, _output) in self.results.items():
            self.assertEqual(code, 0, name)
        self.assertEqual(self.results["audit"][1]["scope"], "locked_training_selection")
        self.assertEqual(self.results["audit"][1]["hparl_train"]["records"],
                         len(read_jsonl(self.root / "data/manifests/subsets/raw/train_hparl_fixture_seed42.jsonl")))
        with self.assertRaises(DataError):  # the full-inventory audit would need the removed audio
            audit.run_audit(self.fresh("full"), workers=1)

    def test_locked_hparl_dev_and_frozen_test_are_reproduced_and_untouched(self):
        lineage = read_json(self.root / LINEAGE)
        self.assertEqual(lineage["status"], "css10_reaudited_mixture_ready")
        unchanged = lineage["locked_inputs_unchanged"]
        self.assertTrue(unchanged["selected_clean_hparl"]["reproduced_byte_identical"])
        self.assertTrue(unchanged["dev"]["reproduced_byte_identical"])
        self.assertEqual(unchanged["frozen_test"]["before"], unchanged["frozen_test"]["after"])
        for relative, content in self.locked.items():
            self.assertEqual((self.root / relative).read_bytes(), content, relative)
        clean = read_json(self.root / "data/clean/selected_train_hparl_fixture_seed42/clean_summary.json")
        drops = clean["subsets"]["hparl_train"]["dropped_by_rule"]
        self.assertEqual(drops, {"annotation_only": 1, "cps_extreme": 1, "overlap_dev_audio": 1,
                                 "overlap_test_audio": 1})

    def test_rerun_is_byte_identical(self):
        outputs = ("data/audit/selected_train_hparl_fixture_seed42", "data/clean/selected_train_hparl_fixture_seed42",
                   LINEAGE, "data/manifests")
        root = self.fresh("rerun")
        first = tree_bytes(root, *outputs)
        for name in ("audit", "clean"):
            self.assertEqual(cli(root, name, "--selection", "configs/training_data_selection.json")[0], 0)
        self.assertEqual(tree_bytes(root, *outputs), first)

    def test_css10_float32_is_measured_with_true_amplitudes(self):
        directory = self.root / "data/audit/selected_train_hparl_fixture_seed42"
        items = [json.loads(line) for line in gzip.decompress(
            (directory / "items_css10_el.jsonl.gz").read_bytes()).decode().splitlines()]
        self.assertTrue(items)
        self.assertEqual({(i["format_tag"], i["encoding"]) for i in items}, {(3, "ieee_float32le")})
        first = next(i for i in items if i["id"].endswith("_0000"))
        spec = next(c for c in CSS10 if c[0].endswith("_0000.wav"))
        samples = float_samples(spec[3], 22050, seed=spec[5])
        expected = 20 * math.log10(math.sqrt(sum(x * x for x in samples) / len(samples)))
        self.assertAlmostEqual(first["rms_dbfs"], expected, places=2)
        summary = read_json(directory / "audit_summary.json")
        self.assertEqual(summary["schema"], "p3-data-audit-v2")
        self.assertIn("format tag", summary["decoder"])
        self.assertEqual(set(summary["environment"]), {"python", "numpy", "system"})

    def test_unchanged_css10_keeps_paths_and_lineage_has_counts_and_overlaps(self):
        lineage = read_json(self.root / LINEAGE)
        self.assertFalse(lineage["css10"]["changed"])
        self.assertEqual(lineage["css10"]["current"]["manifest"]["path"], "data/manifests/css10_el_train.jsonl")
        self.assertFalse(lineage["mixture"]["changed"])
        plan = read_json(self.root / "configs/training_data_selection.json")
        self.assertEqual(lineage["mixture"]["current"]["manifest"], plan["artifacts"]["mixed_manifest"])
        cohort = lineage["mixture"]["cohort"]
        clean = read_jsonl(self.root / plan["artifacts"]["clean_manifest"]["path"])
        self.assertEqual(cohort["records"], len(clean) + lineage["css10"]["kept_records"])
        self.assertEqual(set(cohort["by_source"]), {"hparl", "css10_el"})
        self.assertEqual(cohort["by_source"]["css10_el"]["known_speakers"], 0)
        self.assertEqual(cohort["by_source"]["css10_el"]["records_without_speaker_id"], lineage["css10"]["kept_records"])
        for key in ("literal_test_overlap", "normalized_test_overlap", "test_speakers_also_in_this_training",
                    "sessions_also_in_test"):
            self.assertIn(key, cohort)
        self.assertEqual(cohort["sessions_also_in_test"], 0)

    def test_verify_opens_every_delivered_wav_once_and_fails_on_bad_audio(self):
        code, result = self.results["verify"]
        self.assertEqual(code, 0)
        plan = read_json(self.root / "configs/training_data_selection.json")
        paths = {r["audio_path"] for name in ("raw_manifest", "mixed_manifest")
                 for r in read_jsonl(self.root / plan["artifacts"][name]["path"])}
        paths |= {r["audio_path"] for r in read_jsonl(self.root / "data/manifests/dev.jsonl")}
        self.assertEqual(result["unique_audio_files_checked"], len(paths))
        self.assertIn("ieee_float32le", result["encodings"])
        self.assertTrue(result["test_freeze"]["audio_checked"])
        for damage in ("flip", "pointer"):
            root = self.fresh(f"bad-{damage}")
            wav = root / read_jsonl(root / "data/manifests/css10_el_train.jsonl")[0]["audio_path"]
            data = bytearray(wav.read_bytes())
            if damage == "flip":
                data[-1] ^= 0xFF
            else:
                data = bytearray(b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"0" * 64 + b"\n")
            wav.write_bytes(bytes(data))
            self.assertEqual(cli(root, "verify", "--selection", "configs/training_data_selection.json")[0], 2)

    def test_changed_css10_goes_to_versioned_files_with_old_new_hashes(self):
        root = self.fresh("css10-change")
        css_id = read_jsonl(root / "data/manifests/css10_el_train.jsonl")[0]["id"]
        (root / "configs/review_decisions.jsonl").write_text(json.dumps(
            {"id": css_id, "decision": "drop", "reviewer": "fixture", "date": "2026-11-20",
             "method": "fixture decision (not a real listening)"}) + "\n", encoding="utf-8")
        before = tree_bytes(root, "data/manifests")
        code, result = cli(root, "clean", "--selection", "configs/training_data_selection.json")
        self.assertEqual(code, 0)
        lineage = read_json(root / LINEAGE)
        for part in ("css10", "mixture"):
            self.assertTrue(lineage[part]["changed"])
            current, previous = lineage[part]["current"], lineage[part]["previous"]
            self.assertIn(".css10audit-v2.jsonl", current["manifest"]["path"])
            self.assertNotEqual(current["manifest"]["sha256"], previous["manifest"]["sha256"])
            self.assertEqual((root / previous["manifest"]["path"]).read_bytes(), before[previous["manifest"]["path"]])
        mixed = read_jsonl(root / lineage["mixture"]["current"]["manifest"]["path"])
        self.assertNotIn(css_id, {r["id"] for r in mixed})
        self.assertEqual(lineage["css10"]["dropped_by_rule"], {"duplicate_audio": 1, "human_review": 1})
        code, verified = cli(root, "verify", "--selection", "configs/training_data_selection.json")
        self.assertEqual(code, 0)
        self.assertEqual(verified["current_manifest_source"], LINEAGE)
        self.assertEqual(verified["mixed_hparl_css10"]["manifest"], lineage["mixture"]["current"]["manifest"]["path"])

    def test_locked_hparl_difference_is_reported_and_never_written(self):
        root = self.copy(self.base, "cross-session")
        # session 3 holds byte/PCM copies of session-1 clips: the full inventory removed them,
        # a session-3-only scope cannot see the originals, so recomputation differs.
        write_selection_plan(root, ["olomeleia_3", "olomeleia_8"])
        locked = tree_bytes(root, "data/manifests")
        self.assertEqual(cli(root, "audit", "--selection", "configs/training_data_selection.json")[0], 0)
        self.assertEqual(cli(root, "clean", "--selection", "configs/training_data_selection.json")[0], 2)
        lineage = read_json(root / LINEAGE)
        self.assertEqual(lineage["status"], "locked_hparl_difference_reported")
        check = lineage["locked_inputs_unchanged"]["selected_clean_hparl"]
        self.assertFalse(check["reproduced_byte_identical"])
        extra = {item["id"] for item in check["differences"]["only_in_recomputed"]}
        self.assertEqual(extra, {f"hparl:{TRAIN['dup'][0]}", f"hparl:{TRAIN['pcmdup'][0]}"})
        self.assertEqual(tree_bytes(root, "data/manifests"), locked)

    def test_report_describes_selected_scope_from_outputs(self):
        root = self.fresh("report")
        self.assertEqual(cli(root, "report")[0], 0)
        text = (root / "docs/DATA_REPORT_TR.md").read_text(encoding="utf-8")
        lineage = read_json(root / LINEAGE)
        for expected in ("Güncel kapsam", "kaynak aday envanteridir", "4a. CSS10 IEEE float32", "speaker-disjoint",
                         LINEAGE, "bayt-özdeş yeniden üretildi", "p3-data-audit-v2", "ieee_float32le",
                         "--selection configs/training_data_selection.json",
                         f"{lineage['mixture']['cohort']['records']} kayıt"):
            self.assertIn(expected, text)
        self.assertEqual(cli(root, "report")[0], 0)
        self.assertEqual((root / "docs/DATA_REPORT_TR.md").read_text(encoding="utf-8"), text)

    def test_tampered_locked_artifact_is_refused(self):
        root = self.fresh("tamper")
        path = root / "data/manifests/subsets/train_hparl_fixture_seed42.jsonl"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(DataError, "changed"):
            selection.load_selection(root)
        self.assertEqual(cli(root, "verify", "--selection", "configs/training_data_selection.json")[0], 2)


if __name__ == "__main__":
    unittest.main()
