"""Reproduce Samed's unchanged cleaning on delivered selected RAW HParl audio.

Use an isolated, hard-linked staging root: source/reference/test files in the
repository are never rewritten. Only HParl is audited/cleaned. CSS10 remains a
separate acceptance dependency for the mixed training condition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from greek_asr.data import audit, clean, freeze
from greek_asr.data.common import jsonl_bytes, read_json, read_jsonl, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root, stage, output = args.root.resolve(), args.staging.resolve(), args.output.resolve()
    stage.relative_to(root / "outputs")
    output.relative_to(root)
    if stage.exists() or output.exists():
        raise FileExistsError("choose new staging/output paths; never overwrite acceptance evidence")
    selection = read_json(root / "configs/training_data_selection.json")
    artifacts = selection["artifacts"]
    for item in artifacts.values():
        if sha256_file(root / item["path"]) != item["sha256"]:
            raise ValueError(f"locked training artifact changed: {item['path']}")
    frozen_paths = ["data/manifests/test.freeze.json", "data/manifests/test.jsonl", "data/manifests/references/test.jsonl"]
    before = {path: sha256_file(root / path) for path in frozen_paths}
    stage.mkdir(parents=True)
    report = {"schema": "p3-hparl-training-acceptance-v1", "status": "running",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              "selection_sha256": sha256_file(root / "configs/training_data_selection.json"),
              "scope": "locked HParl RAW train + official dev; unchanged frozen test only hashed",
              "cleaning_implementation": "Samed's original greek_asr.data.audit and clean; no changed rules",
              "css10_accepted": False, "final_test_inference_authorized": False,
              "cleaning_rule_sha256": sha256_file(root / "configs/data_cleaning.json"),
              "implementation_sha256": {str(path.relative_to(root)): sha256_file(path)
                                        for path in sorted((root / "greek_asr/data").glob("*.py"))},
              "frozen_source_before": before}

    def save():
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

    def copy(relative, source=None):
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source or root / relative, target)

    save()
    try:
        copy("configs/data_cleaning.json")
        if (root / "configs/review_decisions.jsonl").exists():
            copy("configs/review_decisions.jsonl")
        copy("data/manifests/raw/hparl_train.jsonl", root / artifacts["raw_manifest"]["path"])
        copy("data/manifests/raw/references/hparl_train.jsonl", root / artifacts["raw_references"]["path"])
        for relative in ["data/manifests/raw/hparl_inventory.json", *frozen_paths]:
            copy(relative)
        for split in ("dev", "test"):
            copy(f"data/manifests/raw/hparl_{split}.jsonl")
            copy(f"data/manifests/raw/references/hparl_{split}.jsonl")
        source_release = root / "data/raw/hparl/release"
        for child in source_release.iterdir():
            if child.name in ("train_wavs", "dev_wavs", "test_wavs"):
                continue
            target = stage / "data/raw/hparl/release" / child.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if child.is_dir():
                shutil.copytree(child, target)
            else:
                shutil.copyfile(child, target)
        if (root / "data/raw/hparl/ACQUISITION.json").exists():
            copy("data/raw/hparl/ACQUISITION.json")
        raw = read_jsonl(stage / "data/manifests/raw/hparl_train.jsonl")
        for split in ("dev", "test"):
            raw.extend(read_jsonl(stage / f"data/manifests/raw/hparl_{split}.jsonl"))
        seen = set()
        for number, row in enumerate(raw, 1):
            source_path = (root / row["audio_path"]).resolve()
            source_path.relative_to(root)
            if sha256_file(source_path) != row["audio_sha256"]:
                raise ValueError(f"missing/LFS pointer/changed original audio: {row['id']}")
            if row["audio_path"] not in seen:
                target = stage / row["audio_path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(source_path, target)
                seen.add(row["audio_path"])
            if number % 8192 == 0:
                print(json.dumps({"phase": "original_audio_verified", "records": number, "total": len(raw)}), flush=True)
        print(json.dumps({"phase": "fresh_hparl_audit", "original_wavs": len(seen)}), flush=True)
        audited = audit.run_audit(stage, workers=args.workers)
        cleaned = clean.run_clean(stage)
        checks = {}
        for name, relative, expected in (
            ("train_audio_manifest", "data/manifests/train.jsonl", artifacts["clean_manifest"]["sha256"]),
            ("train_references", "data/manifests/references/train.jsonl", artifacts["clean_references"]["sha256"]),
            ("dev_audio_manifest", "data/manifests/dev.jsonl", sha256_file(root / "data/manifests/dev.jsonl")),
            ("dev_references", "data/manifests/references/dev.jsonl", sha256_file(root / "data/manifests/references/dev.jsonl"))):
            actual = sha256_file(stage / relative)
            checks[name] = {"expected_sha256": expected, "reproduced_sha256": actual, "matches": actual == expected}
        report["reproduced_artifacts"] = checks
        if not all(item["matches"] for item in checks.values()):
            raise ValueError("fresh used-only cleaning differs from locked training/dev inputs; do not train")
        after = {path: sha256_file(root / path) for path in frozen_paths}
        if after != before:
            raise ValueError("original test metadata changed during verification")
        report.update(status="passed", local_original_audio_verified=True,
                      original_audio_files=len(seen), raw_train_records=selection["cohorts"]["raw_selected_train"]["records"],
                      clean_train_records=selection["cohorts"]["clean_selected_train"]["records"],
                      hparl_only_training_authorized=True,
                      actual_freeze_verification=freeze.verify_freeze(root), frozen_source_after=after,
                      audit_summary_sha256=sha256_file(stage / "data/audit/audit_summary.json"),
                      clean_summary_sha256=sha256_file(stage / "data/clean/clean_summary.json"),
                      cleaning_log_sha256=sha256_file(stage / "data/clean/cleaning_log.jsonl"),
                      fresh_audit_subsets={name: value["records"] for name, value in audited["subsets"].items()},
                      fresh_clean_subsets=cleaned["subsets"])
        print(json.dumps({"status": "passed", "raw_train": report["raw_train_records"],
                          "clean_train": report["clean_train_records"], "all_locked_hashes_match": True}), flush=True)
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error), hparl_only_training_authorized=False)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
