"""End-to-end data pipeline checks on a synthetic corpus with known problems."""
from __future__ import annotations

import contextlib
import gzip
import http.server
import io
import json
import shutil
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path

from greek_asr import infer
from greek_asr.data import audit, clean, freeze, hparl
from greek_asr.data.__main__ import main as data_cli
from greek_asr.data.common import (DataError, kaldi_records, probe_wav_bytes, read_json, read_jsonl, safe_extract,
                                   safe_member_target, sha256_file, validate_manifest_row)
from tests.data_fixtures import CSS10, DEV, TEST, TRAIN, UNLABELLED_TRAIN, make_project, run_stage_one, run_stage_two


def hid(key: str) -> str:
    spec = {**TRAIN, **DEV, **TEST}[key]
    return f"hparl:{spec[0]}"


class PipelineFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = make_project(Path(cls.temporary.name) / "project")
        run_stage_one(cls.root)
        cls.freeze_bytes = (cls.root / "data/manifests/test.freeze.json").read_bytes()
        cls.test_bytes = (cls.root / "data/manifests/test.jsonl").read_bytes()
        run_stage_two(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def copy_project(self) -> Path:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        target = Path(temporary.name) / "project"
        shutil.copytree(self.root, target)
        return target

    def manifest(self, name: str) -> dict[str, dict]:
        return {r["id"]: r for r in read_jsonl(self.root / "data/manifests" / name)}

    def references(self, name: str) -> dict[str, str]:
        return {r["id"]: r["reference"] for r in read_jsonl(self.root / "data/manifests/references" / name)}

    def log(self) -> dict[str, list[dict]]:
        result = {}
        for row in read_jsonl(self.root / "data/clean/cleaning_log.jsonl"):
            result.setdefault(row["id"], []).append(row)
        return result


class RawAcquisitionTests(PipelineFixture):
    def test_raw_record_keeps_source_id_transcript_hash_duration_and_greek(self):
        rows = {r["id"]: r for r in read_jsonl(self.root / "data/manifests/raw/hparl_train.jsonl")}
        refs = {r["id"]: r["reference"] for r in read_jsonl(self.root / "data/manifests/raw/references/hparl_train.jsonl")}
        row = rows[hid("good1")]
        self.assertEqual(row["original_id"], TRAIN["good1"][0])
        self.assertEqual(refs[hid("good1")], "εμείς μπορούμε σήμερα")
        self.assertEqual(refs[hid("noise")], TRAIN["noise"][1])  # raw keeps the annotation tag
        self.assertEqual(row["audio_sha256"], sha256_file(self.root / row["audio_path"]))
        self.assertAlmostEqual(row["duration_seconds"], 2.0)
        self.assertEqual((row["sampling_rate"], row["channels"], row["num_frames"]), (16000, 1, 32000))
        self.assertEqual(row["speaker_id"], "spka")
        self.assertEqual(row["source_recording_id"], "olomeleia_1")
        self.assertTrue(row["audio_path"].startswith("data/raw/hparl/release/train_wavs/"))

    def test_validation_directory_maps_to_dev_and_layout_is_recorded(self):
        layout = read_json(self.root / "data/raw/hparl/LAYOUT.json")
        self.assertEqual(layout["dev"]["metadata_dir"], "hparl_segmented_validation")
        self.assertEqual(layout["dev"]["wav_dirs"], ["validation_wavs"])

    def test_layout_prefers_segmented_metadata_and_records_ignored_dirs(self):
        with tempfile.TemporaryDirectory() as temp:
            release = Path(temp)
            for name in ("hparl_segmented_train", "hparl_train", "hparl_segmented_dev", "hparl_segmented_test",
                         "hparl_all"):
                (release / name).mkdir()
                (release / name / "text").write_text("")
                (release / name / "utt2spk").write_text("")
            for name in ("train_wavs", "dev_wavs", "test_wavs"):
                (release / name).mkdir()
                (release / name / "a.wav").write_bytes(b"")
            layout = hparl.detect_layout(release)
            self.assertEqual(layout["train"]["metadata_dir"], "hparl_segmented_train")
            self.assertEqual(layout["_ignored_metadata_dirs"], ["hparl_all", "hparl_train"])
        utts = {"s-o_1_1_2_1_1": ("x", ""), "s-o_1_3_4_1_1": ("y", "")}
        speakers = {u: ("s", "") for u in utts}
        self_pointing = {u: (f"{u} 0 1", "") for u in utts}
        mapping, method, _ = hparl.recording_ids(utts, speakers, self_pointing)
        self.assertTrue(method.startswith("derived_session"))
        self.assertEqual(set(mapping.values()), {"o_1"})

    def test_source_problems_are_visible_not_silently_dropped(self):
        inventory = read_json(self.root / "data/manifests/raw/hparl_inventory.json")
        issues = {(i["issue"], i["original_id"]) for i in inventory["splits"]["train"]["issues"]}
        self.assertIn(("missing_audio", TRAIN["missing"][0]), issues)
        self.assertIn(("audio_unreadable", TRAIN["corrupt"][0]), issues)
        self.assertIn(("audio_without_transcript", UNLABELLED_TRAIN), issues)
        logged = self.log()
        self.assertEqual(logged[hid("missing")][0]["rule"], "source:missing_audio")
        self.assertEqual(logged[hid("corrupt")][0]["rule"], "source:audio_unreadable")
        self.assertEqual(logged[f"hparl:{UNLABELLED_TRAIN}"][0]["action"], "not_included")

    def test_duplicate_kaldi_ids_and_empty_lines_are_reported(self):
        records, issues = kaldi_records("a x\nb\na y\n".encode(), "text")
        self.assertEqual(records["a"][0], "x")
        self.assertEqual({i["issue"] for i in issues}, {"duplicate_id", "empty_value"})

    def test_hash_mismatch_and_lfs_pointer_are_visible_errors(self):
        row = dict(self.manifest("train.jsonl")[hid("good1")])
        row["audio_sha256"] = "0" * 64
        with self.assertRaisesRegex(DataError, "hash mismatch"):
            validate_manifest_row(row, self.root)
        with self.assertRaisesRegex(DataError, "Git LFS pointer"):
            probe_wav_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:00\nsize 1\n")

    def test_unsafe_archive_paths_and_links_are_rejected(self):
        destination = Path(self.temporary.name) / "extract"
        for name in ("../evil.wav", "/abs.wav", "a/../../evil", "C:/x.wav", "a\\b.wav"):
            with self.assertRaises(DataError):
                safe_member_target(name, destination)
        archive = Path(self.temporary.name) / "slip.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("top/ok.txt", b"ok")
            handle.writestr("top/../../escape.txt", b"bad")
        with self.assertRaises(DataError):
            safe_extract(archive, destination)
        link = Path(self.temporary.name) / "link.zip"
        with zipfile.ZipFile(link, "w") as handle:
            info = zipfile.ZipInfo("top/link")
            info.external_attr = 0o120777 << 16
            handle.writestr(info, "target")
        with self.assertRaisesRegex(DataError, "link"):
            safe_extract(link, destination)


class TestFreezeTests(PipelineFixture):
    def test_test_ids_equal_official_cohort_and_references_are_verbatim(self):
        test = self.manifest("test.jsonl")
        self.assertEqual(set(test), {hid(k) for k in TEST})
        refs = self.references("test.jsonl")
        self.assertEqual(refs[hid("test2")], TEST["test2"][1])  # <spoken_noise> kept in test
        self.assertTrue(all(r["source"] == "hparl" and r["split"] == "test" for r in test.values()))
        freeze_record = read_json(self.root / "data/manifests/test.freeze.json")
        self.assertEqual(freeze_record["record_count"], 3)
        self.assertEqual(freeze_record["test_manifest_sha256"], sha256_file(self.root / "data/manifests/test.jsonl"))

    def test_cleaning_and_rerun_never_change_the_frozen_test(self):
        self.assertEqual((self.root / "data/manifests/test.freeze.json").read_bytes(), self.freeze_bytes)
        self.assertEqual((self.root / "data/manifests/test.jsonl").read_bytes(), self.test_bytes)
        summary = read_json(self.root / "data/clean/clean_summary.json")
        self.assertEqual(summary["test_freeze_before"], summary["test_freeze_after"])
        again = freeze.freeze_test(self.root)
        self.assertTrue(again["already_frozen"])
        self.assertEqual((self.root / "data/manifests/test.freeze.json").read_bytes(), self.freeze_bytes)

    def test_any_change_to_test_audio_or_reference_fails_verification(self):
        root = self.copy_project()
        row = read_jsonl(root / "data/manifests/test.jsonl")[0]
        audio = root / row["audio_path"]
        data = bytearray(audio.read_bytes())
        data[-1] ^= 1
        audio.write_bytes(bytes(data))
        with self.assertRaisesRegex(DataError, "TEST FREEZE VIOLATION"):
            freeze.verify_freeze(root)
        with self.assertRaisesRegex(DataError, "TEST FREEZE VIOLATION"):
            clean.run_clean(root)
        root = self.copy_project()
        path = root / "data/manifests/references/test.jsonl"
        path.write_text(path.read_text(encoding="utf-8").replace("<spoken_noise> ", ""), encoding="utf-8")
        with self.assertRaisesRegex(DataError, "TEST FREEZE VIOLATION"):
            freeze.verify_freeze(root, check_audio=False)

    def test_a_different_freeze_is_never_written_over_the_existing_one(self):
        root = self.copy_project()
        path = root / "data/manifests/raw/references/hparl_test.jsonl"
        path.write_text(path.read_text(encoding="utf-8").replace("ναι", "όχι"), encoding="utf-8")
        with self.assertRaises(DataError):
            freeze.freeze_test(root)

    def test_missing_test_audio_blocks_the_freeze(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = make_project(Path(temporary.name) / "p", drop_test_audio=True)
        with self.assertRaisesRegex(DataError, "official test has 1 source problems"):
            run_stage_one(root)
        self.assertFalse((root / "data/manifests/test.freeze.json").exists())


class CleaningTests(PipelineFixture):
    def test_each_known_problem_gets_the_expected_rule(self):
        logged = self.log()
        expected = {"dup": "duplicate_audio", "pcmdup": "duplicate_pcm", "testcopy": "overlap_test_audio",
                    "devcopy": "overlap_dev_audio", "long": "duration_long", "annot": "annotation_only",
                    "silent": "digital_silence"}
        for key, rule in expected.items():
            entries = logged[hid(key)]
            self.assertEqual([(e["action"], e["rule"]) for e in entries], [("drop", rule)], key)
            self.assertEqual(entries[0]["rule_version"], "p3-data-clean-v1")
            self.assertIsNotNone(entries[0]["before"]["audio_sha256"])
        self.assertEqual(logged[hid("dup")][0]["evidence"], {"duplicate_of": hid("good1")})

    def test_annotation_tag_is_removed_only_from_train_targets_and_logged(self):
        modified = self.log()[hid("noise")][0]
        self.assertEqual(modified["action"], "modify_text")
        self.assertEqual(modified["after_text"], "τους ξεφεύγει η αντικατάσταση της")
        self.assertEqual(modified["before"]["audio_sha256"], modified["after"]["audio_sha256"])
        self.assertEqual(self.references("train.jsonl")[hid("noise")], "τους ξεφεύγει η αντικατάσταση της")
        self.assertEqual(self.references("dev.jsonl")[hid("dev2")], DEV["dev2"][1])

    def test_healthy_records_are_kept_unchanged_and_flags_go_to_review(self):
        train = self.manifest("train.jsonl")
        raw = {r["id"]: r for r in read_jsonl(self.root / "data/manifests/raw/hparl_train.jsonl")}
        for key in ("good1", "good2", "clip"):
            self.assertEqual(train[hid(key)], raw[hid(key)])
        self.assertNotIn(hid("good1"), self.log())
        review = {r["id"]: r for r in read_jsonl(self.root / "data/clean/review_queue.jsonl")}
        self.assertIn("clipping", review[hid("clip")]["flags"])
        self.assertEqual(set(self.manifest("dev.jsonl")), {hid("dev1"), hid("dev2")})

    def test_counts_and_durations_reconcile_with_the_log(self):
        summary = read_json(self.root / "data/clean/clean_summary.json")["subsets"]["hparl_train"]
        raw = read_jsonl(self.root / "data/manifests/raw/hparl_train.jsonl")
        kept = read_jsonl(self.root / "data/manifests/train.jsonl")
        dropped = [e for e in read_jsonl(self.root / "data/clean/cleaning_log.jsonl")
                   if e["subset"] == "hparl_train" and e["action"] == "drop" and not e["rule"].startswith("source:")]
        self.assertEqual(len(raw), len(kept) + len(dropped))
        self.assertAlmostEqual(summary["raw_seconds"] - summary["kept_seconds"],
                               sum(e["before"]["duration_seconds"] for e in dropped), places=3)
        self.assertAlmostEqual(summary["dropped_seconds_by_rule"]["duration_long"], 31.0)

    def test_css10_is_training_only_and_harmonised_with_provenance(self):
        css = self.manifest("css10_el_train.jsonl")
        self.assertTrue(all(r["split"] == "train" and r["source_split"] == "unsplit" for r in css.values()))
        for name in ("dev.jsonl", "test.jsonl", "train.jsonl"):
            self.assertFalse(any(i.startswith("css10_el:") for i in self.manifest(name)), name)
        refs = self.references("css10_el_train.jsonl")
        self.assertEqual(refs["css10_el:Paramythi_horis_onoma_0001"], "είχε τρία παιδιά")
        texts = {r["id"]: r for r in read_jsonl(self.root / "data/manifests/raw/css10_el_texts.jsonl")}
        self.assertEqual(texts["css10_el:Paramythi_horis_onoma_0001"]["original_text"], CSS10[1][1])
        self.assertEqual(self.log()["css10_el:Paramythi_horis_onoma_0002"][0]["rule"], "duplicate_audio")

    def test_hparl_only_and_mixed_conditions_differ_exactly_by_css10(self):
        only, mixed = self.manifest("train.jsonl"), self.manifest("train_hparl_css10.jsonl")
        css = self.manifest("css10_el_train.jsonl")
        self.assertEqual(set(mixed) - set(only), set(css))
        self.assertEqual({k: v for k, v in mixed.items() if k in only}, only)
        conditions = read_json(self.root / "data/manifests/conditions.json")
        self.assertEqual(conditions["conditions"]["hparl_css10"]["records"], {"css10_el": 2, "hparl": 4})
        self.assertEqual(conditions["shared_dev"]["manifest"], "data/manifests/dev.jsonl")

    def test_rerun_with_same_inputs_is_byte_identical(self):
        outputs = ["data/manifests/train.jsonl", "data/manifests/train_hparl_css10.jsonl", "data/manifests/dev.jsonl",
                   "data/manifests/references/train.jsonl", "data/clean/cleaning_log.jsonl",
                   "data/clean/review_queue.jsonl", "data/manifests/conditions.json",
                   "data/audit/items_hparl_train.jsonl.gz", "data/audit/audit_summary.json", "docs/DATA_REPORT_TR.md"]
        root = self.copy_project()
        before = {name: (root / name).read_bytes() for name in outputs}
        audit.run_audit(root, workers=2)
        clean.run_clean(root)
        from greek_asr.data import report
        report.write_report(root)
        for name in outputs:
            self.assertEqual((root / name).read_bytes(), before[name], name)

    def test_stale_audit_is_refused(self):
        root = self.copy_project()
        config = root / "configs/data_cleaning.json"
        data = json.loads(config.read_text(encoding="utf-8"))
        data["flags"]["duration_long_seconds"] = 20.0
        config.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(DataError, "stale"):
            clean.run_clean(root)

    def test_human_review_must_be_attributable_and_never_touch_test(self):
        root = self.copy_project()
        decisions = root / "configs/review_decisions.jsonl"
        decisions.write_text(json.dumps({"id": hid("clip"), "decision": "drop", "reviewer": "SmtOrtakaya",
                                         "date": "2026-11-20", "method": "listened with headphones",
                                         "note": "square-wave tone, no speech"}) + "\n", encoding="utf-8")
        clean.run_clean(root)
        entry = [e for e in read_jsonl(root / "data/clean/cleaning_log.jsonl") if e["id"] == hid("clip")][0]
        self.assertEqual((entry["action"], entry["rule"]), ("drop", "human_review"))
        self.assertEqual(entry["evidence"]["reviewer"], "SmtOrtakaya")
        decisions.write_text(json.dumps({"id": hid("test1"), "decision": "drop", "reviewer": "x", "date": "d",
                                         "method": "m"}) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(DataError, "frozen test"):
            clean.run_clean(root)
        decisions.write_text(json.dumps({"id": hid("clip"), "decision": "drop"}) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(DataError, "attributable"):
            clean.run_clean(root)

    def test_audit_items_hold_measurements_not_text(self):
        rows = [json.loads(line) for line in gzip.decompress(
            (self.root / "data/audit/items_hparl_train.jsonl.gz").read_bytes()).decode().splitlines()]
        self.assertTrue(rows)
        self.assertFalse(any("reference" in row for row in rows))
        silent = next(r for r in rows if r["id"] == hid("silent"))
        self.assertEqual(silent["speech_fraction"], 0.0)


class ContractAndCliTests(PipelineFixture):
    def test_shared_runner_accepts_every_delivered_manifest(self):
        for name, split in (("train.jsonl", "train"), ("train_hparl_css10.jsonl", "train"), ("dev.jsonl", "dev"),
                            ("test.jsonl", "test")):
            records = infer.load_manifest(self.root / "data/manifests" / name, self.root, split)
            self.assertTrue(records, name)

    def test_paths_are_portable_relative_posix(self):
        for name in ("train.jsonl", "dev.jsonl", "test.jsonl", "train_hparl_css10.jsonl"):
            for row in read_jsonl(self.root / "data/manifests" / name):
                self.assertFalse(row["audio_path"].startswith("/") or ":" in row["audio_path"] or "\\" in row["audio_path"])
        moved = self.copy_project()
        self.assertEqual(freeze.verify_freeze(moved)["status"], "ok")

    def test_cli_verify_and_subset(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(data_cli(["--root", str(self.root), "verify"]), 0)
        self.assertEqual(json.loads(output.getvalue())["test_freeze"]["status"], "ok")
        root = self.copy_project()
        first = clean.select_subset(root, hours=0.001, seed=7)
        second = clean.select_subset(root, hours=0.001, seed=7)
        self.assertEqual(first, second)
        rows = read_jsonl(root / "data/manifests/subsets/train_hparl_0.001h_seed7.jsonl")
        sessions = {r["source_recording_id"] for r in rows}
        full = read_jsonl(root / "data/manifests/train.jsonl")
        self.assertEqual({r["id"] for r in full if r["source_recording_id"] in sessions}, {r["id"] for r in rows})

    def test_release_shards_round_trip_and_detect_tampering(self):
        from greek_asr.data import publish

        root = self.copy_project()
        shards = Path(tempfile.mkdtemp(dir=Path(root).parent))
        result = publish.pack(root, "data/raw/hparl/release/train_wavs", shards, max_bytes=1_200_000)
        self.assertGreater(len(result["shards"]), 1)
        index = root / "data/release_index/data_raw_hparl_release_train_wavs.json"
        originals = {p: p.read_bytes() for p in (root / "data/raw/hparl/release/train_wavs").iterdir()}
        shutil.rmtree(root / "data/raw/hparl/release/train_wavs")
        self.assertEqual(publish.unpack(root, index, shards)["restored_files"], len(originals))
        self.assertEqual({p: p.read_bytes() for p in originals}, originals)
        first = shards / result["shards"][0]["file"]
        first.write_bytes(first.read_bytes()[:-1] + b"\1")
        with self.assertRaisesRegex(DataError, "corrupted"):
            publish.unpack(root, index, shards)

    def test_report_is_generated_from_outputs(self):
        report = (self.root / "docs/DATA_REPORT_TR.md").read_text(encoding="utf-8")
        freeze_record = read_json(self.root / "data/manifests/test.freeze.json")
        self.assertIn(freeze_record["test_manifest_sha256"], report)
        self.assertIn("`duration_long`", report)
        self.assertIn("τους ξεφεύγει η αντικατάσταση της", report)


class DownloadTests(unittest.TestCase):
    def serve(self, payload: bytes, honour_range: bool = True):
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                start = int(self.headers["Range"].split("=")[1].split("-")[0]) if honour_range else 0
                body = payload[start:start + 1000]  # short responses force resumption
                self.send_response(206 if honour_range else 200)
                self.send_header("Content-Range", f"bytes {start}-{start + len(body) - 1}/{len(payload)}")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_address[1]}/"

    def test_resumes_from_partial_file_and_hashes_complete_archive(self):
        payload = bytes(range(256)) * 20
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "hparl.zip"
            Path(temp, "hparl.zip.part").write_bytes(payload[:777])
            record = hparl.download(target, url=self.serve(payload), total=len(payload), headers={}, log=lambda *_: None)
            self.assertEqual(target.read_bytes(), payload)
            self.assertEqual(record["archive_sha256"], sha256_file(target))
            self.assertFalse(Path(temp, "hparl.zip.part").exists())

    def test_server_ignoring_range_is_rejected(self):
        payload = b"x" * 3000
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "hparl.zip"
            Path(temp, "hparl.zip.part").write_bytes(payload[:10])
            with self.assertRaisesRegex(DataError, "byte range"):
                hparl.download(target, url=self.serve(payload, honour_range=False), total=len(payload), headers={},
                               log=lambda *_: None)
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
