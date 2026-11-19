"""CSS10 Greek (Kaggle v1) acquisition and raw training-only manifest.

The Kaggle archive needs a signed-in download, so the user supplies the ZIP.
Its bytes are hashed, extracted safely and the producer's ``transcript.txt``
(``audio|original text|normalized text|duration``) is preserved verbatim.
"""
from __future__ import annotations

import datetime as _dt
import json
import urllib.request
from collections import Counter
from pathlib import Path

from .common import (DataError, ROOT, contract_row, json_bytes, jsonl_bytes, keep_bytes, probe_wav, read_json,
                     relative_posix, safe_extract, sha256_bytes, sha256_file, write_output)

KAGGLE_REF = "bryanpark/greek-single-speaker-speech-dataset"
KAGGLE_METADATA = f"https://www.kaggle.com/api/v1/datasets/view/{KAGGLE_REF}"
DATASET = {"name": "CSS10 Greek (Greek Single Speaker Speech Dataset)", "kaggle_ref": KAGGLE_REF,
           "kaggle_dataset_id": 30096, "planned_version": 1, "license_declared_by_distributor": "CC0: Public Domain",
           "producer_repository": "https://github.com/Kyubyong/css10",
           "paper": "https://arxiv.org/abs/1903.11269",
           "audiobook": "https://librivox.org/paramythi-horis-onoma-by-penelope-delta/"}
RAW = ROOT / "data/raw/css10_el"
MANIFESTS = ROOT / "data/manifests/raw"


def fetch_metadata(opener=urllib.request.urlopen) -> tuple[bytes | None, str | None]:
    try:
        with opener(KAGGLE_METADATA, timeout=30) as response:
            data = response.read(4 * 1024 * 1024 + 1)
        json.loads(data)
        return data, None
    except Exception as error:  # recorded in provenance, never hidden
        return None, f"{type(error).__name__}: {error}"


def acquire(archive: Path, raw: Path = RAW, metadata: bytes | None = None, metadata_error: str | None = None) -> dict:
    archive = Path(archive)
    if not archive.is_file():
        raise DataError(f"CSS10 archive not found: {archive}")
    record = {"dataset": DATASET, "archive_file": archive.name, "archive_bytes": archive.stat().st_size,
              "archive_sha256": sha256_file(archive),
              "acquired_utc": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()}
    raw.mkdir(parents=True, exist_ok=True)
    if metadata is not None:
        parsed = json.loads(metadata)
        keep_bytes(raw / "kaggle_metadata.json", metadata)
        record["kaggle_metadata_sha256"] = sha256_bytes(metadata)
        record["kaggle_metadata_fields"] = {key: parsed.get(key) for key in
                                            ("currentVersionNumber", "licenseName", "totalBytes", "id", "ref")}
    else:
        record["kaggle_metadata_error"] = metadata_error
    previous_path = raw / "ACQUISITION.json"
    if previous_path.exists():
        previous = read_json(previous_path)
        if previous.get("archive_sha256") != record["archive_sha256"]:
            raise DataError("a different CSS10 archive was already acquired; refusing to mix releases")
    else:
        keep_bytes(previous_path, json_bytes(record))
    result = safe_extract(archive, raw / "release")
    listing = "member\tpath\tbytes\tcrc32\tsha256\n" + "".join(
        f"{m['member']}\t{m['path']}\t{m['bytes']}\t{m['crc32']}\t{m['sha256']}\n" for m in result["members"])
    write_output(raw / "archive_members.tsv", listing.encode("utf-8"))
    return {**record, "extracted_files": len(result["members"])}


def locate_transcript(release: Path, override: str | None = None) -> Path:
    if override:
        path = (release / override).resolve()
        if not path.is_relative_to(release.resolve()) or not path.is_file():
            raise DataError(f"transcript override not found inside the release: {override}")
        return path
    found = sorted(release.rglob("transcript.txt"))
    if len(found) != 1:
        raise DataError(f"expected exactly one transcript.txt, found {[p.relative_to(release).as_posix() for p in found]}; "
                        "pass --transcript with the intended relative path")
    return found[0]


def build_raw_manifest(raw: Path = RAW, manifests: Path = MANIFESTS, root: Path = ROOT,
                       transcript_override: str | None = None) -> dict:
    release = raw / "release"
    if not release.is_dir():
        raise DataError("run `css10-acquire` first")
    transcript = locate_transcript(release, transcript_override)
    data = transcript.read_bytes()
    try:
        lines = data.decode("utf-8").splitlines(keepends=True)
    except UnicodeDecodeError as error:
        raise DataError(f"CSS10 transcript is not UTF-8: {error}") from None
    rows, references, texts, issues, seen_paths, seen_ids = [], [], [], [], set(), set()
    for number, line in enumerate(lines, 1):
        stripped = line.rstrip("\r\n")
        if not stripped.strip():
            continue
        fields = stripped.split("|")
        if len(fields) != 4:
            issues.append({"issue": "malformed_transcript_line", "line": number, "fields": len(fields)})
            continue
        relative_audio, original_text, normalized_text, reported = fields
        audio_path = (transcript.parent / relative_audio).resolve()
        if not audio_path.is_relative_to(release.resolve()):
            issues.append({"issue": "audio_path_escapes_release", "line": number, "path": relative_audio})
            continue
        original_id = Path(relative_audio).stem
        if original_id in seen_ids or audio_path in seen_paths:
            issues.append({"issue": "duplicate_id", "line": number, "original_id": original_id})
            continue
        seen_ids.add(original_id)
        seen_paths.add(audio_path)
        try:
            reported_seconds = float(reported)
        except ValueError:
            reported_seconds = None
            issues.append({"issue": "unparseable_reported_duration", "line": number, "original_id": original_id})
        if not audio_path.is_file():
            issues.append({"issue": "missing_audio", "line": number, "original_id": original_id})
            continue
        try:
            audio = probe_wav(audio_path)
        except DataError as error:
            issues.append({"issue": "audio_unreadable", "original_id": original_id, "detail": str(error)})
            continue
        if audio["num_frames"] == 0:
            issues.append({"issue": "zero_frames", "original_id": original_id})
            continue
        if not normalized_text.strip():
            issues.append({"issue": "empty_value", "original_id": original_id})
        identity = f"css10_el:{original_id}"
        rows.append(contract_row(
            id=identity, source="css10_el", original_id=original_id, source_split="unsplit", split="train",
            audio_path=relative_posix(audio_path, root), audio_sha256=audio["audio_sha256"],
            sampling_rate=audio["sampling_rate"], channels=audio["channels"], num_frames=audio["num_frames"],
            duration_seconds=audio["duration_seconds"], source_recording_id=None, speaker_id=None))
        references.append({"id": identity, "reference": normalized_text})
        texts.append({"id": identity, "original_text": original_text, "normalized_text": normalized_text,
                      "reported_duration_seconds": reported_seconds, "transcript_line": number,
                      "transcript_line_sha256": sha256_bytes(line.encode("utf-8"))})
    referenced = {path for path in seen_paths}
    unreferenced = sorted(relative_posix(p, root) for p in release.rglob("*")
                          if p.is_file() and p.suffix.lower() == ".wav" and p.resolve() not in referenced)
    order = sorted(range(len(rows)), key=lambda i: rows[i]["id"])
    rows, references, texts = ([seq[i] for i in order] for seq in (rows, references, texts))
    inventory = {
        "dataset": DATASET, "transcript": relative_posix(transcript, root), "transcript_sha256": sha256_bytes(data),
        "transcript_lines": sum(1 for line in lines if line.strip()), "manifest_records": len(rows),
        "audio_seconds": sum(row["duration_seconds"] for row in rows),
        "sampling_rates": dict(Counter(str(row["sampling_rate"]) for row in rows)),
        "channels": dict(Counter(str(row["channels"]) for row in rows)),
        "speaker_note": "Producer reports one reader (Rapunzelina); no per-utterance speaker field is supplied, "
                        "so speaker_id stays null.",
        "unreferenced_wav_files": len(unreferenced), "unreferenced_wav_examples": unreferenced[:20],
        "issues": issues, "issue_counts": dict(sorted(Counter(item["issue"] for item in issues).items())),
        "manifest_sha256": write_output(manifests / "css10_el.jsonl", jsonl_bytes(rows)),
        "references_sha256": write_output(manifests / "references" / "css10_el.jsonl", jsonl_bytes(references)),
        "texts_sha256": write_output(manifests / "css10_el_texts.jsonl", jsonl_bytes(texts)),
    }
    acquisition = raw / "ACQUISITION.json"
    if acquisition.exists():
        inventory["archive_sha256"] = read_json(acquisition)["archive_sha256"]
    write_output(manifests / "css10_el_inventory.json", json_bytes(inventory))
    return inventory
