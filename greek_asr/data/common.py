"""Shared, dependency-light helpers for the Project 3 data pipeline.

Everything written here is deterministic: JSON keys are sorted, rows are
written in a caller-defined order and no timestamps are added implicitly.
Existing files with different bytes are never silently replaced unless the
caller explicitly asks for a regenerable output.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import unicodedata
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator

ROOT = Path(__file__).resolve().parents[2]

CONTRACT_FIELDS = (
    "id", "source", "original_id", "source_split", "split", "audio_path", "audio_sha256",
    "sampling_rate", "channels", "num_frames", "duration_seconds", "source_recording_id", "speaker_id",
)
REQUIRED_FIELDS = CONTRACT_FIELDS[:11]
SOURCES = ("hparl", "css10_el")
SPLITS = ("train", "dev", "test")
TAG_PATTERN = re.compile(r"<[^<>\r\n]+>")
CHUNK = 1024 * 1024


class DataError(ValueError):
    """A visible data/provenance problem; never swallowed silently."""


# ---------------------------------------------------------------- hashing / IO

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dumps(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=indent, allow_nan=False)


def json_bytes(value: Any) -> bytes:
    return (json_dumps(value, indent=2) + "\n").encode("utf-8")


def jsonl_bytes(rows: Iterable[dict]) -> bytes:
    return b"".join((json_dumps(row) + "\n").encode("utf-8") for row in rows)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise DataError(f"{path}:{number}: invalid JSON ({error.msg})") from None
        if not isinstance(row, dict):
            raise DataError(f"{path}:{number}: each JSONL row must be an object")
        rows.append(row)
    return rows


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_output(path: Path, data: bytes) -> str:
    """Write a regenerable output atomically and return its SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == data:
        return sha256_bytes(data)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)
    return sha256_bytes(data)


def keep_bytes(path: Path, data: bytes) -> str:
    """Write source/immutable bytes once; refuse to replace different bytes."""
    path = Path(path)
    if path.exists():
        if path.read_bytes() != data:
            raise DataError(f"refusing to replace different existing bytes: {path}")
        return sha256_bytes(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
    return sha256_bytes(data)


def relative_posix(path: Path, root: Path) -> str:
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def resolve_inside(root: Path, relative: str) -> Path:
    """Resolve a manifest path; reject absolute, drive-letter or escaping paths."""
    if not isinstance(relative, str) or not relative:
        raise DataError("audio_path must be a non-empty project-relative string")
    if relative.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", relative) or "\\" in relative:
        raise DataError(f"machine-specific or non-POSIX path is not allowed: {relative}")
    target = (Path(root) / relative).resolve()
    if not target.is_relative_to(Path(root).resolve()):
        raise DataError(f"path escapes the project root: {relative}")
    return target


# ------------------------------------------------------------------- archives

def safe_member_target(name: str, destination: Path, strip_prefix: str = "") -> Path | None:
    """Map a ZIP member to a path inside destination, or reject it.

    Absolute paths, drive letters, backslashes, ``..`` components and empty
    names are rejected rather than repaired. Returns ``None`` for directories.
    """
    if not name or name.startswith("/") or "\\" in name or re.match(r"^[A-Za-z]:", name):
        raise DataError(f"unsafe archive member path: {name!r}")
    parts = PurePosixPath(name).parts
    if any(part in ("..", "") for part in parts):
        raise DataError(f"unsafe archive member path: {name!r}")
    if name.endswith("/"):
        return None
    if strip_prefix:
        if not name.startswith(strip_prefix):
            raise DataError(f"archive member outside expected top-level directory: {name!r}")
        name = name[len(strip_prefix):]
        if not name:
            return None
    target = (Path(destination) / name).resolve()
    if not target.is_relative_to(Path(destination).resolve()):
        raise DataError(f"archive member escapes destination: {name!r}")
    return target


def zip_member_is_link(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0o177777
    return stat.S_ISLNK(mode)


def common_top_level(names: Iterable[str]) -> str:
    """Return ``"top/"`` when every member shares one top-level directory."""
    tops = {name.split("/", 1)[0] for name in names if name}
    if len(tops) == 1:
        top = tops.pop()
        if all("/" in name or name == top + "/" for name in names if name):
            return top + "/"
    return ""


def safe_extract(archive_path: Path, destination: Path, *, strip_top_level: bool = True) -> dict:
    """Extract every member with CRC checks; existing identical files are reused.

    Links, unsafe paths, duplicate member names and differing existing bytes are
    visible errors. Returns a deterministic member inventory.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    inventory, seen = [], set()
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        duplicates = sorted({name for name in names if names.count(name) > 1}) if len(set(names)) != len(names) else []
        if duplicates:
            raise DataError(f"archive contains duplicate member names: {duplicates[:5]}")
        prefix = common_top_level(names) if strip_top_level else ""
        for info in infos:
            if zip_member_is_link(info):
                raise DataError(f"archive member is a link and is rejected: {info.filename!r}")
            target = safe_member_target(info.filename, destination, prefix)
            if target is None or info.is_dir():
                continue
            if target in seen:
                raise DataError(f"two archive members map to one path: {info.filename!r}")
            seen.add(target)
            digest = hashlib.sha256()
            if target.exists():
                with archive.open(info) as source:  # zipfile verifies CRC32 at EOF
                    for chunk in iter(lambda: source.read(CHUNK), b""):
                        digest.update(chunk)
                if target.stat().st_size != info.file_size or sha256_file(target) != digest.hexdigest():
                    raise DataError(f"refusing to replace different existing bytes: {target}")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + ".partial")
                with archive.open(info) as source, temporary.open("wb") as sink:
                    for chunk in iter(lambda: source.read(CHUNK), b""):
                        digest.update(chunk)
                        sink.write(chunk)
                os.replace(temporary, target)
            inventory.append({"member": info.filename, "path": relative_posix(target, destination),
                              "bytes": info.file_size, "crc32": f"{info.CRC:08x}",
                              "sha256": digest.hexdigest()})
    inventory.sort(key=lambda row: row["member"])
    return {"strip_prefix": prefix, "members": inventory}


# ---------------------------------------------------------------------- audio

def probe_wav(path: Path) -> dict:
    """Return measured WAV properties; raise DataError for unusable audio."""
    try:
        data = Path(path).read_bytes()
    except FileNotFoundError:
        raise DataError(f"missing audio file: {path}") from None
    return probe_wav_bytes(data, str(path))


LFS_POINTER = b"version https://git-lfs.github.com/spec/"


ENCODINGS = {(1, 2): "pcm_s16le", (1, 4): "pcm_s32le", (3, 4): "ieee_float32le"}


def probe_wav_bytes(data: bytes, label: str = "audio") -> dict:
    """Measure one original WAV with the shared RIFF reader (greek_asr/audio.py).

    The RIFF format tag, not the 4-byte sample width alone, separates integer
    PCM32 from IEEE float32. Bytes are only read, never converted. A Git LFS
    pointer, a truncated/malformed file, an unsupported encoding and NaN/Inf
    float samples are visible DataErrors. ``pcm_sha256`` hashes the exact
    data-chunk bytes (the same bytes the previous ``wave`` probe hashed).
    """
    from ..audio import read_wav_bytes

    if data.startswith(LFS_POINTER):
        raise DataError(f"Git LFS pointer instead of audio bytes (run `git lfs pull`): {label}")
    try:
        audio = read_wav_bytes(data, label)
    except ValueError as error:
        raise DataError(f"undecodable WAV ({error})") from None
    encoding = ENCODINGS.get((audio.format_tag, audio.sample_width_bytes))
    if encoding is None:  # read_wav_bytes already restricts this; keep the data contract explicit
        raise DataError(f"unsupported WAV encoding (format tag {audio.format_tag}, "
                        f"{audio.sample_width_bytes * 8}-bit): {label}")
    if audio.format_tag == 3:
        import numpy as np  # data-pipeline dependency, see requirements-data.txt

        if not np.isfinite(np.frombuffer(audio.payload, dtype="<f4")).all():
            raise DataError(f"IEEE float WAV contains NaN/Inf samples: {label}")
    return {"sampling_rate": audio.sampling_rate, "channels": audio.channels,
            "sample_width_bytes": audio.sample_width_bytes, "num_frames": audio.num_frames,
            "format_tag": audio.format_tag, "encoding": encoding,
            "duration_seconds": audio.num_frames / audio.sampling_rate,
            "audio_bytes": len(data), "audio_sha256": sha256_bytes(data),
            "pcm_sha256": sha256_bytes(audio.payload), "_payload": audio.payload}


def validate_manifest_row(row: dict, root: Path, *, check_audio: bool = True, cache: dict | None = None) -> None:
    """Mirror greek_asr.infer.load_manifest so data outputs are runner-ready."""
    missing = [field for field in REQUIRED_FIELDS if field not in row]
    if missing:
        raise DataError(f"{row.get('id')}: missing manifest fields {missing}")
    if row["source"] not in SOURCES:
        raise DataError(f"{row['id']}: unsupported source {row['source']!r}")
    if not isinstance(row["original_id"], str) or row["id"] != f"{row['source']}:{row['original_id']}":
        raise DataError(f"{row['id']}: ID must be source:original_id")
    if row["split"] not in SPLITS or row["source_split"] not in (*SPLITS, "unsplit"):
        raise DataError(f"{row['id']}: invalid split")
    if row["source"] == "css10_el" and row["split"] != "train":
        raise DataError(f"{row['id']}: CSS10 is training-only")
    if row["source"] == "hparl" and row["source_split"] != row["split"]:
        raise DataError(f"{row['id']}: HParl official split must be preserved")
    for field in ("sampling_rate", "channels", "num_frames"):
        if type(row[field]) is not int or row[field] <= 0:
            raise DataError(f"{row['id']}: invalid {field}")
    duration = row["duration_seconds"]
    if (type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0
            or abs(duration - row["num_frames"] / row["sampling_rate"]) > 0.001):
        raise DataError(f"{row['id']}: invalid duration")
    if not isinstance(row["audio_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", row["audio_sha256"]):
        raise DataError(f"{row['id']}: invalid audio_sha256")
    path = resolve_inside(root, row["audio_path"])
    if check_audio:
        if not path.is_file():
            raise DataError(f"{row['id']}: audio file missing: {row['audio_path']}")
        measured = cache.get(row["audio_path"]) if cache is not None else None
        if measured is None:
            measured = {k: v for k, v in probe_wav(path).items() if k != "_payload"}
            if cache is not None:
                cache[row["audio_path"]] = measured
        if measured["audio_sha256"] != row["audio_sha256"]:
            raise DataError(f"{row['id']}: audio hash mismatch")
        for field in ("sampling_rate", "channels", "num_frames"):
            if measured[field] != row[field]:
                raise DataError(f"{row['id']}: WAV header disagrees with manifest field {field}")


def contract_row(**values: Any) -> dict:
    row = {field: values.get(field) for field in CONTRACT_FIELDS}
    extra = set(values) - set(CONTRACT_FIELDS)
    if extra:
        raise DataError(f"unexpected manifest fields: {sorted(extra)}")
    return row


# ----------------------------------------------------------------------- text

def kaldi_records(data: bytes, label: str) -> tuple[dict[str, tuple[str, str]], list[dict]]:
    """Parse ``<id><whitespace><value>`` lines keeping the original line bytes.

    Returns ``{id: (value, original_line)}`` plus visible issues (duplicate IDs,
    lines without a value). The value is everything after the first whitespace
    run, unchanged (no normalization).
    """
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise DataError(f"{label}: not valid UTF-8 ({error})") from None
    records, issues = {}, []
    for number, line in enumerate(text.splitlines(keepends=True), 1):
        stripped = line.rstrip("\r\n")
        if not stripped.strip():
            continue
        fields = stripped.split(maxsplit=1)
        identity = fields[0]
        value = fields[1] if len(fields) == 2 else ""
        if identity in records:
            issues.append({"issue": "duplicate_id", "file": label, "line": number, "original_id": identity})
            continue
        if not value.strip():
            issues.append({"issue": "empty_value", "file": label, "line": number, "original_id": identity})
        records[identity] = (value, line)
    return records, issues


def annotation_tags(text: str) -> list[str]:
    return sorted(set(TAG_PATTERN.findall(text)))


def strip_annotation_tags(text: str) -> str:
    return " ".join(TAG_PATTERN.sub(" ", text).split())


def is_greek_letter(character: str) -> bool:
    return character.isalpha() and "GREEK" in unicodedata.name(character, "")


def character_inventory(texts: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for text in texts:
        for character in unicodedata.normalize("NFC", text):
            counts[character] = counts.get(character, 0) + 1
    return dict(sorted(counts.items()))


def hms(seconds: float) -> str:
    total = int(round(seconds))
    return f"{total // 3600:d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def iter_chunks(items: list, size: int) -> Iterator[list]:
    for start in range(0, len(items), size):
        yield items[start:start + size]
