"""Read-only integration review of a teammate's committed ASR data delivery.

This computes speaker/session overlap and literal repeated test sentences from
manifests. It never changes the frozen test, prepares data or runs a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_stats(rows: list[dict], delivered_files: set[str]) -> dict:
    return {
        "utterances": len(rows),
        "hours": sum(r["duration_seconds"] for r in rows) / 3600,
        "speakers_with_ids": len({r["speaker_id"] for r in rows if r.get("speaker_id")}),
        "utterances_without_speaker_id": sum(not r.get("speaker_id") for r in rows),
        "recordings_or_sessions_with_ids": len({r["source_recording_id"] for r in rows
                                                if r.get("source_recording_id")}),
        "utterances_without_recording_id": sum(not r.get("source_recording_id") for r in rows),
        "audio_paths_present_in_checkout": sum(r["audio_path"] in delivered_files for r in rows),
    }


def literal_overlap(train_refs: list[dict], test_refs: list[dict]) -> dict:
    training = {r["reference"] for r in train_refs}
    repeated = [r for r in test_refs if r["reference"] in training]
    return {"policy": "exact Unicode string equality; no casefold, tag removal or whitespace change",
            "test_utterances_repeated_in_training": len(repeated),
            "distinct_repeated_test_sentences": len({r["reference"] for r in repeated}),
            "test_utterances_total": len(test_refs),
            "matching_test_ids": [r["id"] for r in repeated]}


def review(root: Path) -> dict:
    root = root.resolve()
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    base = root / "data/manifests"
    raw = {split: read_rows(base / f"raw/hparl_{split}.jsonl") for split in ("train", "dev", "test")}
    clean = {name: read_rows(base / f"{name}.jsonl")
             for name in ("train", "dev", "test", "css10_el_train", "train_hparl_css10")}
    raw_css = read_rows(base / "raw/css10_el.jsonl")
    raw_refs = {split: read_rows(base / f"raw/references/hparl_{split}.jsonl")
                for split in ("train", "dev", "test")}
    clean_refs = {name: read_rows(base / f"references/{name}.jsonl")
                  for name in ("train", "train_hparl_css10")}
    delivered = {p.relative_to(root).as_posix() for p in (root / "data").rglob("*") if p.is_file()}
    speakers = {split: {r["speaker_id"] for r in rows if r.get("speaker_id")}
                for split, rows in raw.items()}
    sessions = {split: {r["source_recording_id"] for r in rows if r.get("source_recording_id")}
                for split, rows in raw.items()}
    pairs = (("train", "dev"), ("train", "test"), ("dev", "test"))
    freeze_path = base / "test.freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8-sig"))
    freeze_hash_checks = {
        "manifest_matches_freeze": digest(base / "test.jsonl") == freeze["test_manifest_sha256"],
        "references_match_freeze": digest(base / "references/test.jsonl") == freeze["references_sha256"],
        "frozen_manifest_equals_raw_manifest": (base / "test.jsonl").read_bytes() ==
                                               (base / "raw/hparl_test.jsonl").read_bytes(),
        "frozen_references_equal_raw_references": (base / "references/test.jsonl").read_bytes() ==
                                                  (base / "raw/references/hparl_test.jsonl").read_bytes(),
        "ids_match_freeze": [r["id"] for r in clean["test"]] == [r["id"] for r in freeze["records"]],
    }
    log_path = root / "data/clean/cleaning_log.jsonl"
    log = read_rows(log_path)
    cps_drops = [r for r in log if r.get("action") == "drop" and r.get("rule") == "cps_implausible"]
    clean_summary_path = root / "data/clean/clean_summary.json"
    summary = json.loads(clean_summary_path.read_text(encoding="utf-8-sig"))
    first_freeze_commit = subprocess.check_output(
        ["git", "log", "--diff-filter=A", "--format=%H", "--", "data/manifests/test.freeze.json"],
        cwd=root, text=True).splitlines()[-1]
    freeze_history_checks = {}
    for relative in ("data/manifests/test.freeze.json", "data/manifests/test.jsonl",
                     "data/manifests/references/test.jsonl"):
        original = subprocess.check_output(["git", "show", first_freeze_commit + ":" + relative], cwd=root)
        freeze_history_checks[relative] = hashlib.sha256(original).hexdigest() == digest(root / relative)
    # Import the reviewed report generator only after all direct file checks.
    # build_report returns a string; it does not write to the reviewed checkout.
    import importlib.util
    import sys
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("greek_asr.data.report", root / "greek_asr/data/report.py")
    report_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report_module)
    regenerated_report = report_module.build_report(root).encode("utf-8")
    input_files = [*base.glob("*.jsonl"), *base.glob("references/*.jsonl"),
                   *base.glob("raw/*.jsonl"), *base.glob("raw/references/*.jsonl"),
                   freeze_path, log_path, clean_summary_path, root / "configs/data_cleaning.json"]
    return {
        "schema": "p3-data-review-v1", "reviewed_commit": source_commit,
        "scope": "Metadata/committed-code review; file presence is checked but full audio bytes are not rehashed",
        "raw_splits": {k: manifest_stats(v, delivered) for k, v in raw.items()} |
                      {"css10_el": manifest_stats(raw_css, delivered)},
        "clean_conditions": {k: manifest_stats(v, delivered) for k, v in clean.items()},
        "speaker_overlaps": {f"{a}&{b}": len(speakers[a] & speakers[b]) for a, b in pairs},
        "session_overlaps": {f"{a}&{b}": len(sessions[a] & sessions[b]) for a, b in pairs},
        "test_utterances_with_training_speaker": sum(r.get("speaker_id") in speakers["train"] for r in raw["test"]),
        "literal_sentence_overlap": {
            "raw_hparl_train_vs_original_test": literal_overlap(raw_refs["train"], raw_refs["test"]),
            **{name: literal_overlap(refs, raw_refs["test"]) for name, refs in clean_refs.items()},
        },
        "freeze_metadata_checks": freeze_hash_checks,
        "first_freeze_commit": first_freeze_commit,
        "unchanged_since_initial_freeze_commit": freeze_history_checks,
        "generated_data_report_equals_committed": regenerated_report == (root / "docs/DATA_REPORT_TR.md").read_bytes(),
        "generated_data_report_sha256": hashlib.sha256(regenerated_report).hexdigest(),
        "full_audio_freeze_independently_verified": False,
        "missing_original_test_metadata": [path for path in freeze["source"]["metadata_files_sha256"]
            if not (root / "data/raw/hparl/release" / path).is_file()],
        "cleaning_log_hash_matches_summary": digest(log_path) == summary["log_sha256"],
        "drop_rules": dict(Counter(r["rule"] for r in log if r.get("action") == "drop")),
        "cps_drops": [{"id": r["id"], "subset": r["subset"], "evidence": r["evidence"]} for r in cps_drops],
        "human_review_decisions_applied": summary["review_decisions_applied"],
        "inputs_sha256": {p.relative_to(root).as_posix(): digest(p) for p in sorted(set(input_files))},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = review(args.data_root)
    payload = json.dumps(result, ensure_ascii=True, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        brief = {k: result[k] for k in ("reviewed_commit", "speaker_overlaps", "session_overlaps",
                 "test_utterances_with_training_speaker", "freeze_metadata_checks", "drop_rules")}
        brief["literal_overlap_counts"] = {k: {f: v[f] for f in
            ("test_utterances_repeated_in_training", "distinct_repeated_test_sentences")}
            for k, v in result["literal_sentence_overlap"].items()}
        print(json.dumps(brief, ensure_ascii=True, indent=2))
    else:
        print(payload)


if __name__ == "__main__":
    main()
