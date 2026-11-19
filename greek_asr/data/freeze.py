"""Early, immutable freeze of the original HParl test cohort.

The freeze copies the raw test manifest and verbatim references, then records
source identity and per-record hashes. Nothing about the test is cleaned,
filtered, re-split, resampled or relabelled. An existing freeze is never
overwritten with different content; ``verify_freeze`` recomputes every hash.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

from .common import (LFS_POINTER, DataError, ROOT, json_bytes, jsonl_bytes, kaldi_records, keep_bytes, read_json, read_jsonl,
                     resolve_inside, sha256_bytes, sha256_file, validate_manifest_row)

SCHEMA = "hparl-test-freeze-v1"
BLOCKING_ISSUES = {"missing_audio", "ambiguous_audio", "audio_unreadable", "zero_frames", "duplicate_id"}
POLICY = ("Original HParl 1.0.0 test split, frozen before any train/dev cleaning or corpus mixing. "
          "Audio bytes and source transcript lines are unchanged; no record was added, removed, "
          "re-split, resampled, trimmed, normalised or relabelled. CSS10 is excluded.")


def paths(root: Path) -> dict:
    return {"manifest": root / "data/manifests/test.jsonl",
            "references": root / "data/manifests/references/test.jsonl",
            "freeze": root / "data/manifests/test.freeze.json",
            "raw_manifest": root / "data/manifests/raw/hparl_test.jsonl",
            "raw_references": root / "data/manifests/raw/references/hparl_test.jsonl",
            "inventory": root / "data/manifests/raw/hparl_inventory.json",
            "acquisition": root / "data/raw/hparl/ACQUISITION.json",
            "release": root / "data/raw/hparl/release"}


def _head(path: Path) -> bytes:
    with path.open("rb") as stream:
        return stream.read(len(LFS_POINTER))


def _ids_sha(ids: list[str]) -> str:
    return sha256_bytes("\n".join(ids).encode("utf-8"))


def build_freeze_content(root: Path = ROOT) -> tuple[dict, bytes, bytes]:
    p = paths(root)
    inventory = read_json(p["inventory"])
    test_inventory = inventory["splits"]["test"]
    blocking = [issue for issue in test_inventory["issues"] if issue["issue"] in BLOCKING_ISSUES]
    if blocking:
        raise DataError(f"official test has {len(blocking)} source problems that would change its membership "
                        f"(e.g. {blocking[:3]}); stop and report them to Serdar/Ozan instead of freezing")
    rows = read_jsonl(p["raw_manifest"])
    references = read_jsonl(p["raw_references"])
    if [r["id"] for r in rows] != [r["id"] for r in references]:
        raise DataError("raw test manifest and references disagree on IDs/order")
    if any(row["source"] != "hparl" or row["split"] != "test" or row["source_split"] != "test" for row in rows):
        raise DataError("raw test manifest contains a non-HParl or non-test record")
    metadata_dir = inventory["layout"]["test"]["metadata_dir"]
    text_bytes = (p["release"] / metadata_dir / "text").read_bytes()
    if sha256_bytes(text_bytes) != test_inventory["metadata_files_sha256"][f"{metadata_dir}/text"]:
        raise DataError("test Kaldi text changed since the raw manifests were built")
    source_text, _ = kaldi_records(text_bytes, "test/text")
    if sorted(source_text) != [row["original_id"] for row in rows]:
        raise DataError("test manifest IDs are not exactly the official test transcript IDs")
    records = []
    for row, reference in zip(rows, references):
        validate_manifest_row(row, root)
        value, line = source_text[row["original_id"]]
        if reference["reference"] != value:
            raise DataError(f"{row['id']}: reference is not the verbatim source transcript")
        audio = resolve_inside(root, row["audio_path"])
        records.append({"id": row["id"], "original_id": row["original_id"], "audio_path": row["audio_path"],
                        "audio_bytes": audio.stat().st_size, "audio_sha256": row["audio_sha256"],
                        "duration_seconds": row["duration_seconds"],
                        "reference_sha256": sha256_bytes(value.encode("utf-8")),
                        "reference_line_sha256": sha256_bytes(line.encode("utf-8")),
                        "source_recording_id": row["source_recording_id"], "speaker_id": row["speaker_id"]})
    manifest_bytes, reference_bytes = jsonl_bytes(rows), jsonl_bytes(references)
    acquisition = read_json(p["acquisition"]) if p["acquisition"].exists() else {}
    ids = [row["id"] for row in rows]
    content = {
        "schema": SCHEMA, "status": "frozen", "policy": POLICY,
        "source": {"dataset": inventory["dataset"], "archive_sha256": acquisition.get("archive_sha256"),
                   "archive_bytes": acquisition.get("archive_bytes"),
                   "official_split": "test", "metadata_dir": metadata_dir,
                   "wav_dirs": inventory["layout"]["test"]["wav_dirs"],
                   "metadata_files_sha256": test_inventory["metadata_files_sha256"],
                   "recording_id_method": test_inventory["recording_id_method"]},
        "record_count": len(rows), "total_duration_seconds": sum(r["duration_seconds"] for r in rows),
        "speakers": len({r["speaker_id"] for r in rows if r["speaker_id"]}),
        "recordings_or_sessions": len({r["source_recording_id"] for r in rows if r["source_recording_id"]}),
        "ids_sha256": _ids_sha(ids),
        "test_manifest_path": "data/manifests/test.jsonl", "test_manifest_sha256": sha256_bytes(manifest_bytes),
        "references_path": "data/manifests/references/test.jsonl", "references_sha256": sha256_bytes(reference_bytes),
        "source_notes": [issue for issue in test_inventory["issues"] if issue["issue"] not in BLOCKING_ISSUES],
        "records": records,
    }
    return content, manifest_bytes, reference_bytes


def freeze_test(root: Path = ROOT, now: str | None = None) -> dict:
    p = paths(root)
    content, manifest_bytes, reference_bytes = build_freeze_content(root)
    if p["freeze"].exists():
        existing = read_json(p["freeze"])
        created = existing.pop("created_utc", None)
        if existing != content:
            raise DataError("a different test freeze already exists; the frozen test must not be rewritten")
        keep_bytes(p["manifest"], manifest_bytes)
        keep_bytes(p["references"], reference_bytes)
        return {**content, "created_utc": created, "already_frozen": True}
    keep_bytes(p["manifest"], manifest_bytes)
    keep_bytes(p["references"], reference_bytes)
    created = now or _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()
    keep_bytes(p["freeze"], json_bytes({**content, "created_utc": created}))
    return {**content, "created_utc": created, "already_frozen": False}


def verify_freeze(root: Path = ROOT, *, check_audio: bool = True) -> dict:
    """Recompute the frozen cohort; any difference is a visible failure."""
    p = paths(root)
    if not p["freeze"].exists():
        raise DataError("test.freeze.json is missing; run `freeze-test` before any train/dev cleaning")
    freeze = read_json(p["freeze"])
    problems = []
    if freeze.get("schema") != SCHEMA or freeze.get("status") != "frozen":
        problems.append("unexpected freeze schema/status")
    for key, path in (("test_manifest_sha256", p["manifest"]), ("references_sha256", p["references"])):
        if not path.exists() or sha256_file(path) != freeze.get(key):
            problems.append(f"{path.relative_to(root).as_posix()} differs from the frozen hash")
    rows = read_jsonl(p["manifest"]) if p["manifest"].exists() else []
    references = {r["id"]: r["reference"] for r in read_jsonl(p["references"])} if p["references"].exists() else {}
    frozen = freeze.get("records", [])
    if [r["id"] for r in rows] != [r["id"] for r in frozen] or _ids_sha([r["id"] for r in rows]) != freeze.get("ids_sha256"):
        problems.append("test ID membership/order differs from the freeze")
    if len(rows) != freeze.get("record_count"):
        problems.append("test record count differs from the freeze")
    if any(row.get("source") != "hparl" or row.get("split") != "test" for row in rows):
        problems.append("test manifest contains a non-HParl or non-test record")
    for row, record in zip(rows, frozen):
        if row["audio_sha256"] != record["audio_sha256"] or row["audio_path"] != record["audio_path"]:
            problems.append(f"{row['id']}: audio identity differs from the freeze")
        if sha256_bytes(references.get(row["id"], "\0").encode("utf-8")) != record["reference_sha256"]:
            problems.append(f"{row['id']}: reference differs from the freeze")
        if check_audio:
            try:
                audio = resolve_inside(root, row["audio_path"])
                if audio.is_file() and _head(audio) == LFS_POINTER:
                    problems.append(f"{row['id']}: Git LFS pointer, not audio (run `git lfs pull`)")
                elif not audio.is_file() or audio.stat().st_size != record["audio_bytes"] or sha256_file(audio) != record["audio_sha256"]:
                    problems.append(f"{row['id']}: audio bytes differ from the freeze")
            except DataError as error:
                problems.append(f"{row['id']}: {error}")
        if len(problems) > 50:
            break
    release = p["release"]
    for relative, digest in freeze.get("source", {}).get("metadata_files_sha256", {}).items():
        source = release / relative
        if not source.is_file() or sha256_file(source) != digest:
            problems.append(f"source test metadata missing or changed: {relative}")
    if problems:
        raise DataError("TEST FREEZE VIOLATION: " + "; ".join(problems[:20]))
    return {"status": "ok", "records": len(rows), "test_manifest_sha256": freeze["test_manifest_sha256"],
            "references_sha256": freeze["references_sha256"], "ids_sha256": freeze["ids_sha256"],
            "audio_checked": check_audio, "freeze_sha256": sha256_file(p["freeze"])}
