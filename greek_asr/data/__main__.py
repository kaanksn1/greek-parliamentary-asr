"""Command line for the Project 3 data pipeline (Samed, data owner).

Order matters: the original test is frozen before any train/dev cleaning or
CSS10 mixing. ``prepare-freeze`` and ``prepare-clean`` run the two stages.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import audit, clean, css10, freeze, hparl, publish, report
from .common import ROOT, DataError, read_json, read_jsonl, validate_manifest_row


def _print(value) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str))


def cmd_hparl_download(args) -> dict:
    archive = args.root / "data/downloads/hparl.zip" if args.archive is None else args.archive
    record = hparl.download(archive)
    catalog, error = hparl.fetch_catalog()
    hparl.record_acquisition(record, args.root / "data/raw/hparl", catalog, error)
    return record


def cmd_hparl_inspect(args) -> dict:
    return hparl.inspect_archive(args.archive or args.root / "data/downloads/hparl.zip")


def cmd_hparl_extract(args) -> dict:
    return hparl.extract(args.archive or args.root / "data/downloads/hparl.zip", args.root / "data/raw/hparl")


def cmd_hparl_manifests(args) -> dict:
    layout = read_json(args.layout) if args.layout else None
    inventory = hparl.build_raw_manifests(args.root / "data/raw/hparl", args.root / "data/manifests/raw",
                                          args.root, layout)
    return {split: {k: v for k, v in data.items() if k != "issues"} for split, data in inventory["splits"].items()}


def cmd_freeze(args) -> dict:
    result = freeze.freeze_test(args.root)
    return {k: v for k, v in result.items() if k != "records"}


def cmd_verify_freeze(args) -> dict:
    return freeze.verify_freeze(args.root, check_audio=not args.skip_audio)


def cmd_css10_acquire(args) -> dict:
    metadata, error = (None, "metadata fetch disabled")
    if args.metadata_json:
        metadata, error = Path(args.metadata_json).read_bytes(), None
    elif not args.no_fetch_metadata:
        metadata, error = css10.fetch_metadata()
    return css10.acquire(args.archive, args.root / "data/raw/css10_el", metadata, error)


def cmd_css10_manifest(args) -> dict:
    inventory = css10.build_raw_manifest(args.root / "data/raw/css10_el", args.root / "data/manifests/raw",
                                         args.root, args.transcript)
    return {k: v for k, v in inventory.items() if k not in ("issues",)}


def cmd_audit(args) -> dict:
    freeze.verify_freeze(args.root, check_audio=False)
    summary = audit.run_audit(args.root, workers=args.workers)
    return {subset: {"records": data["records"], "hours": round(data["seconds"] / 3600, 3),
                     "flag_counts": data["flag_counts"]} for subset, data in summary["subsets"].items()}


def cmd_clean(args) -> dict:
    summary = clean.run_clean(args.root)
    return {k: v for k, v in summary.items() if k in ("subsets", "log_entries", "review_queue_records")}


def cmd_subset(args) -> dict:
    result = clean.select_subset(args.root, args.hours, args.seed, not args.no_css10)
    return {k: v for k, v in result.items() if k != "group_keys"}


def cmd_report(args) -> dict:
    path = report.write_report(args.root)
    return {"report": path.relative_to(args.root).as_posix()}


def cmd_verify(args) -> dict:
    """Validate every delivered manifest against the shared contract and real audio."""
    root = args.root
    result = {"test_freeze": freeze.verify_freeze(root)}
    files = ["train.jsonl", "dev.jsonl", "train_hparl_css10.jsonl", "css10_el_train.jsonl", "test.jsonl"]
    for name in files:
        path = root / "data/manifests" / name
        if not path.exists():
            result[name] = "missing"
            continue
        rows = read_jsonl(path)
        references = read_jsonl(root / "data/manifests/references" / name)
        if [r["id"] for r in rows] != [r["id"] for r in references]:
            raise DataError(f"{name}: references do not match manifest IDs/order")
        if len({r["id"] for r in rows}) != len(rows):
            raise DataError(f"{name}: duplicate IDs")
        for row in rows:
            if "reference" in row or "text" in row:
                raise DataError(f"{name}: gold text must not appear in an audio manifest")
            validate_manifest_row(row, root, check_audio=not args.skip_audio)
        result[name] = {"records": len(rows), "hours": round(sum(r["duration_seconds"] for r in rows) / 3600, 3)}
    test_ids = {r["id"] for r in read_jsonl(root / "data/manifests/test.jsonl")}
    for name in files[:-1]:
        path = root / "data/manifests" / name
        if path.exists() and test_ids & {r["id"] for r in read_jsonl(path)}:
            raise DataError(f"{name} shares IDs with the frozen test")
    return result


def cmd_publish_plan(args) -> dict:
    return publish.plan(args.root, args.quota_gib)


def cmd_pack(args) -> dict:
    return publish.pack(args.root, args.source, args.output, args.max_bytes)


def cmd_unpack(args) -> dict:
    return publish.unpack(args.root, args.index, args.shards)


def cmd_prepare_freeze(args) -> dict:
    steps = {"extract": cmd_hparl_extract(args), "manifests": cmd_hparl_manifests(args), "freeze": cmd_freeze(args),
             "verify_freeze": cmd_verify_freeze(args)}
    return steps


def cmd_prepare_clean(args) -> dict:
    steps = {"verify_freeze": freeze.verify_freeze(args.root)}
    if args.css10_archive:
        args.archive = args.css10_archive
        steps["css10_acquire"] = cmd_css10_acquire(args)
    if (args.root / "data/raw/css10_el/release").is_dir():
        steps["css10_manifest"] = cmd_css10_manifest(args)
    steps["audit"] = cmd_audit(args)
    steps["clean"] = cmd_clean(args)
    steps["report"] = cmd_report(args)
    args.skip_audio = False
    steps["verify"] = cmd_verify(args)
    return steps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m greek_asr.data", description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name, handler, help_text):
        command = sub.add_parser(name, help=help_text)
        command.set_defaults(handler=handler)
        return command

    add("hparl-download", cmd_hparl_download, "resumable download of the official HParl ZIP (~11.9 GB)") \
        .add_argument("--archive", type=Path)
    add("hparl-inspect", cmd_hparl_inspect, "summarise the downloaded ZIP's directories").add_argument("--archive", type=Path)
    add("hparl-extract", cmd_hparl_extract, "CRC-checked safe extraction into data/raw/hparl/release") \
        .add_argument("--archive", type=Path)
    add("hparl-manifests", cmd_hparl_manifests, "raw contract manifests + verbatim references for all splits") \
        .add_argument("--layout", type=Path, help="explicit split layout JSON if auto-detection fails")
    add("freeze-test", cmd_freeze, "write data/manifests/test.jsonl, references and test.freeze.json")
    add("verify-freeze", cmd_verify_freeze, "recompute every frozen test hash") \
        .add_argument("--skip-audio", action="store_true")
    css = add("css10-acquire", cmd_css10_acquire, "hash and safely extract the Kaggle CSS10 Greek ZIP")
    css.add_argument("--archive", type=Path, required=True)
    css.add_argument("--metadata-json", type=Path)
    css.add_argument("--no-fetch-metadata", action="store_true")
    add("css10-manifest", cmd_css10_manifest, "raw CSS10 manifest (training only)") \
        .add_argument("--transcript", help="relative path of transcript.txt inside the release, if ambiguous")
    add("audit", cmd_audit, "measure quality/duplicates/overlaps (no data changes)") \
        .add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    add("clean", cmd_clean, "apply the versioned cleaning policy and write clean manifests + log")
    subset = add("subset", cmd_subset, "seeded whole-session subset of the clean HParl train")
    subset.add_argument("--hours", type=float, required=True)
    subset.add_argument("--seed", type=int, default=42)
    subset.add_argument("--no-css10", action="store_true")
    add("report", cmd_report, "generate docs/DATA_REPORT_TR.md from the recorded outputs")
    add("verify", cmd_verify, "validate all delivered manifests against the contract and audio") \
        .add_argument("--skip-audio", action="store_true")
    add("publish-plan", cmd_publish_plan, "sizes of raw/clean deliverables vs a Git LFS quota") \
        .add_argument("--quota-gib", type=float, default=10.0)
    packer = add("pack", cmd_pack, "deterministic tar shards (< GitHub release asset limit) + committed index")
    packer.add_argument("--source", required=True, help="project-relative directory, e.g. data/raw/hparl/release/train_wavs")
    packer.add_argument("--output", type=Path, required=True, help="directory outside the repo for the shards")
    packer.add_argument("--max-bytes", type=int, default=1_900_000_000)
    unpacker = add("unpack", cmd_unpack, "verify downloaded shards and restore files")
    unpacker.add_argument("--index", type=Path, required=True)
    unpacker.add_argument("--shards", type=Path, required=True)
    stage1 = add("prepare-freeze", cmd_prepare_freeze, "stage 1: extract, raw manifests, test freeze, verify")
    stage1.add_argument("--archive", type=Path)
    stage1.add_argument("--layout", type=Path)
    stage1.set_defaults(skip_audio=False)
    stage2 = add("prepare-clean", cmd_prepare_clean, "stage 2: CSS10, audit, clean, report, verify")
    stage2.add_argument("--css10-archive", type=Path)
    stage2.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    stage2.add_argument("--transcript")
    stage2.add_argument("--no-fetch-metadata", action="store_true")
    stage2.set_defaults(metadata_json=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Greek text must not crash a Windows console
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = build_parser().parse_args(argv)
    args.root = args.root.resolve()
    try:
        _print(args.handler(args))
    except DataError as error:
        print(f"DATA ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
