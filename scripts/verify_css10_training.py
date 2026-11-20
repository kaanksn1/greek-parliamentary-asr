"""Independently reproduce the submitted CSS10 pipeline in isolated staging.

Uses the data owner's unchanged selected-scope commands; never rewrites source
audio, frozen test or the submitted audit/clean reports. No ASR/test predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    staging = (root / args.staging).resolve()
    output = (root / args.output).resolve()
    staging.relative_to(root / "outputs")
    output.relative_to(root / "outputs")
    if staging.exists() or output.exists():
        raise FileExistsError("use fresh staging/output; preserve previous acceptance runs")
    sys.path.insert(0, str(root))
    from greek_asr.data.common import read_jsonl

    selection_path = "configs/training_data_selection.json"
    lineage_path = "configs/css10_mixture_lineage.json"
    plan = read(root / selection_path)
    submitted = read(root / lineage_path)
    if submitted["status"] != "css10_reaudited_mixture_ready" or submitted["audit"]["schema"] != "p3-data-audit-v2":
        raise ValueError("the submitted format-aware CSS10 lineage is not ready")
    locked = {identity["path"] for identity in plan["artifacts"].values()}
    locked.update((selection_path, "data/manifests/dev.jsonl", "data/manifests/references/dev.jsonl",
                   "data/manifests/test.jsonl", "data/manifests/references/test.jsonl", "data/manifests/test.freeze.json"))
    before = {relative: digest(root / relative) for relative in sorted(locked)}
    primary_config = read(root / "configs/sequence/whisper_hparl_only.json")
    mixed_config = read(root / "configs/sequence/whisper_hparl_css10.json")
    condition_fields = {"condition", "train_manifest", "train_references", "_comment"}
    if {k: v for k, v in primary_config.items() if k not in condition_fields} != {
            k: v for k, v in mixed_config.items() if k not in condition_fields}:
        raise ValueError("the paired Whisper optimization settings differ")
    started = time.monotonic()
    staging.mkdir(parents=True)
    for relative in ("configs", "data/manifests", "data/audit", "data/clean"):
        shutil.copytree(root / relative, staging / relative)
    for relative in ("data/raw/hparl/release", "data/raw/css10_el/release"):
        shutil.copytree(root / relative, staging / relative,
                        ignore=lambda directory, names: [name for name in names if name.lower().endswith(".wav")])
    rows = read_jsonl(root / plan["artifacts"]["raw_manifest"]["path"])
    for relative in ("data/manifests/dev.jsonl", "data/manifests/test.jsonl", "data/manifests/css10_el_train.jsonl"):
        rows += read_jsonl(root / relative)
    paths = {row["audio_path"] for row in rows}
    for relative in sorted(paths):
        source, target = (root / relative).resolve(), (staging / relative).resolve()
        source.relative_to(root)
        target.relative_to(staging)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, target)
    print(json.dumps({"staging_original_audio_paths": len(paths)}), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, HSA_ENABLE_DXG_DETECTION="0", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    commands = [
        ["verify-source-metadata"],
        ["audit", "--selection", selection_path, "--workers", "4"],
        ["clean", "--selection", selection_path],
        ["verify", "--selection", selection_path],
    ]
    records = []
    for number, arguments in enumerate(commands, 1):
        command = [sys.executable, "-m", "greek_asr.data", "--root", str(staging), *arguments]
        begin = time.monotonic()
        result = subprocess.run(command, cwd=root, env=environment, capture_output=True, text=True, encoding="utf-8")
        log = output.parent / f"css10_check_{number:02d}_{arguments[0]}.log"
        log.write_text(result.stdout + "\nSTDERR\n" + result.stderr, encoding="utf-8")
        records.append({"arguments": arguments, "exit_code": result.returncode,
                        "seconds": time.monotonic() - begin, "log": log.relative_to(root).as_posix()})
        print(json.dumps(records[-1]), flush=True)
        if result.returncode:
            raise RuntimeError(f"independent {arguments[0]} failed; inspect {log}")
        if arguments[0] == "verify":
            verification = json.loads(result.stdout)
    reproduced = read(staging / lineage_path)
    if reproduced["status"] != "css10_reaudited_mixture_ready":
        raise ValueError("reproduced CSS10 lineage did not pass")
    for label in ("selected_clean_hparl", "dev"):
        if not reproduced["locked_inputs_unchanged"][label]["reproduced_byte_identical"]:
            raise ValueError(f"locked {label} did not reproduce")
    for label in ("css10", "mixture"):
        if reproduced[label]["current"] != submitted[label]["current"] or reproduced[label]["changed"]:
            raise ValueError(f"reproduced {label} differs from the submitted locked condition")
    if reproduced["css10"]["signal_audit"]["current"] != submitted["css10"]["signal_audit"]["current"]:
        raise ValueError("reproduced real CSS10 signal statistics/flags differ")
    if reproduced["clean"]["cleaning_log"]["sha256"] != submitted["clean"]["cleaning_log"]["sha256"]:
        raise ValueError("reproduced cleaning decisions differ from the submission")
    if reproduced["clean"]["review_queue"]["sha256"] != submitted["clean"]["review_queue"]["sha256"]:
        raise ValueError("reproduced review queue differs from the submission")
    for relative, expected in before.items():
        if digest(root / relative) != expected or digest(staging / relative) != expected:
            raise ValueError(f"locked input changed: {relative}")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    result = {"schema": "p3-css10-training-acceptance-v1", "status": "passed",
              "created_utc": datetime.now(timezone.utc).isoformat(), "data_code_commit": commit,
              "owner_verifier_sha256": digest(Path(__file__)), "selection_sha256": digest(root / selection_path),
              "submitted_lineage_sha256": digest(root / lineage_path), "mixed_training_authorized": True,
              "paired_whisper_settings_identical": True, "optimization_settings": {
                  k: v for k, v in mixed_config.items() if k not in condition_fields},
              "original_audio_paths": len(paths), "audio_verification": verification,
              "reproduced_artifacts": {
                  "dev_audio_manifest": {"expected_sha256": before["data/manifests/dev.jsonl"]},
                  "dev_references": {"expected_sha256": before["data/manifests/references/dev.jsonl"]},
                  "mixed_manifest": reproduced["mixture"]["current"]["manifest"],
                  "mixed_references": reproduced["mixture"]["current"]["references"]},
              "frozen_source_before": {key: value for key, value in before.items() if key in (
                  "data/manifests/test.jsonl", "data/manifests/references/test.jsonl", "data/manifests/test.freeze.json")},
              "reproduced_css10_signal_audit": reproduced["css10"]["signal_audit"]["current"],
              "cleaning_log_sha256": reproduced["clean"]["cleaning_log"]["sha256"],
              "review_queue_sha256": reproduced["clean"]["review_queue"]["sha256"],
              "reproduced_lineage": (staging / lineage_path).relative_to(root).as_posix(),
              "commands": records, "elapsed_seconds": time.monotonic() - started,
              "test_predictions_generated": False, "final_test_inference_authorized": False}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "mixed_training_authorized": True,
                      "original_audio_paths": len(paths), "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
