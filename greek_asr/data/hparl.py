"""HParl 1.0.0 acquisition, safe extraction and raw (uncleaned) manifests.

Acquisition keeps the producer bytes: the downloaded ZIP is hashed, every
member is CRC-checked during extraction and extracted files are never
rewritten. Raw manifests describe every official utterance that has a
readable WAV; anything that cannot enter a manifest is listed as a visible
issue in the inventory instead of being dropped silently.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
import time
import urllib.error
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from .common import (DataError, ROOT, contract_row, json_bytes, jsonl_bytes, kaldi_records, keep_bytes,
                     probe_wav, read_json, relative_posix, safe_extract, sha256_bytes, sha256_file,
                     write_output)

URL = "https://inventory.clarin.gr/storage/"
CATALOG = "https://inventory.clarin.gr/catalogue_backend/api/registry/metadatarecord/1602/"
ARCHIVE_BYTES = 11887118613
HEADERS = {
    "ELG-RESOURCE-ID": "1602", "ELG-RESOURCE-DISTRIBUTION-ID": "1024",
    "filename": "1b1c180978d84429b8f0df18895d946c/2e192bae4738437a8e22519f9fc25817/hparl.zip",
    "scope": "private", "Accept-Encoding": "identity",
}
DATASET = {"name": "HParl: Hellenic Parliamentary Speech Corpus", "version": "1.0.0",
           "license": "CC-BY-NC-4.0", "record": "https://inventory.clarin.gr/corpus/1602",
           "persistent_identifier": "http://hdl.handle.net/11500/CLARIN-EL-0000-0000-7665-A"}

DOWNLOAD = ROOT / "data/downloads/hparl.zip"
RAW = ROOT / "data/raw/hparl"
MANIFESTS = ROOT / "data/manifests/raw"

SPLIT_TOKENS = {"train": "train", "dev": "dev", "development": "dev", "valid": "dev",
                "validation": "dev", "test": "test"}
METADATA_FILES = ("text", "utt2spk", "spk2utt", "wav.scp", "segments", "reco2dur", "utt2dur",
                  "spk2gender", "feats.scp", "cmvn.scp")
ID_PATTERN = re.compile(r"^(?P<session>.+)_(?P<start>\d+)_(?P<end>\d+)_(?P<part>\d+)_(?P<sub>\d+)$")


# ------------------------------------------------------------------ download

def download(destination: Path = DOWNLOAD, *, url: str = URL, total: int = ARCHIVE_BYTES,
             headers: dict | None = None, retries: int = 20, chunk: int = 8 * 1024 * 1024,
             opener=urllib.request.urlopen, log=print) -> dict:
    """Resumable bounded HTTP download into ``<destination>.part``.

    Each request asks for ``bytes=<offset>-``; the server must answer 206 with
    the exact ``Content-Range`` start and total. Only a complete, size-checked
    file is renamed to its final name. An existing complete file is reused.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    headers = dict(HEADERS if headers is None else headers)
    if destination.exists():
        if destination.stat().st_size != total:
            raise DataError(f"{destination} exists with an unexpected size; move it aside and retry")
    else:
        attempt = 0
        while True:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > total:
                raise DataError(f"{partial} is larger than the source archive; delete it and retry")
            if offset == total:
                break
            request = urllib.request.Request(url, headers={**headers, "Range": f"bytes={offset}-"})
            try:
                with opener(request, timeout=60) as response:
                    content_range = response.headers.get("Content-Range", "")
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                    if response.status != 206 or not match or int(match[1]) != offset or int(match[3]) != total:
                        raise DataError(f"server did not honour the requested byte range: "
                                        f"status={response.status} Content-Range={content_range!r}")
                    with partial.open("ab") as sink:
                        reported = offset
                        for block in iter(lambda: response.read(chunk), b""):
                            sink.write(block)
                            position = sink.tell()
                            if position > total:
                                raise DataError("server sent more bytes than the archive size")
                            if position - reported >= 256 * 1024 * 1024:
                                reported = position
                                log(json.dumps({"downloaded_gib": round(position / 1024 ** 3, 2),
                                                "total_gib": round(total / 1024 ** 3, 2)}))
                attempt = 0
                log(json.dumps({"downloaded_bytes": partial.stat().st_size, "total_bytes": total}))
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
                attempt += 1
                if attempt > retries:
                    raise DataError(f"download failed after {retries} retries at byte {offset}: {error}") from None
                log(json.dumps({"retry": attempt, "offset": offset, "error": str(error)}))
                time.sleep(min(60, 2 ** attempt))
        os.replace(partial, destination)
    record = {"dataset": DATASET, "archive_url": url, "request_headers": {k: v for k, v in headers.items()
                                                                          if k != "Accept-Encoding"},
              "archive_bytes": destination.stat().st_size, "archive_sha256": sha256_file(destination),
              "archive_file": destination.name,
              "downloaded_utc": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()}
    return record


def fetch_catalog(opener=urllib.request.urlopen) -> tuple[bytes | None, str | None]:
    try:
        with opener(CATALOG, timeout=30) as response:
            data = response.read(4 * 1024 * 1024 + 1)
        if len(data) > 4 * 1024 * 1024:
            return None, "catalog record unexpectedly large"
        json.loads(data)
        return data, None
    except Exception as error:  # recorded, not hidden
        return None, f"{type(error).__name__}: {error}"


def record_acquisition(record: dict, raw: Path = RAW, catalog: bytes | None = None,
                       catalog_error: str | None = None) -> Path:
    raw.mkdir(parents=True, exist_ok=True)
    if catalog is not None:
        keep_bytes(raw / "catalog_record_1602.json", catalog)
        record = {**record, "catalog_record_sha256": sha256_bytes(catalog)}
    else:
        record = {**record, "catalog_record_error": catalog_error}
    path = raw / "ACQUISITION.json"
    if path.exists():
        previous = read_json(path)
        if previous.get("archive_sha256") != record["archive_sha256"]:
            raise DataError("a different HParl archive was already acquired; refusing to mix releases")
        return path
    keep_bytes(path, json_bytes(record))
    return path


# ------------------------------------------------------------------- archive

def inspect_archive(archive: Path) -> dict:
    """Summarise directories of the official ZIP without extracting it."""
    directories: dict[str, Counter] = defaultdict(Counter)
    with zipfile.ZipFile(archive) as handle:
        infos = handle.infolist()
        for info in infos:
            if info.is_dir():
                continue
            parent, _, name = info.filename.rpartition("/")
            kind = "wav" if name.lower().endswith(".wav") else name
            directories[parent][kind] += 1
    return {"members": len(infos),
            "directories": {key: dict(sorted(value.items())) for key, value in sorted(directories.items())}}


def extract(archive: Path = DOWNLOAD, raw: Path = RAW) -> dict:
    acquisition = raw / "ACQUISITION.json"
    if not acquisition.exists():
        raise DataError("run `hparl-download` first; acquisition provenance is missing")
    expected = read_json(acquisition)
    if archive.stat().st_size != expected["archive_bytes"]:
        raise DataError("archive size differs from the recorded acquisition")
    result = safe_extract(archive, raw / "release")
    members = result["members"]
    listing = "member\tpath\tbytes\tcrc32\tsha256\n" + "".join(
        f"{m['member']}\t{m['path']}\t{m['bytes']}\t{m['crc32']}\t{m['sha256']}\n" for m in members)
    write_output(raw / "archive_members.tsv", listing.encode("utf-8"))
    summary = {"archive_sha256": expected["archive_sha256"], "strip_prefix": result["strip_prefix"],
               "extracted_to": "release/", "files": len(members), "bytes": sum(m["bytes"] for m in members),
               "wav_files": sum(m["member"].lower().endswith(".wav") for m in members),
               "members_listing_sha256": sha256_bytes(listing.encode("utf-8"))}
    write_output(raw / "EXTRACTION.json", json_bytes(summary))
    return summary


# -------------------------------------------------------------------- layout

def _split_of(name: str) -> str | None:
    tokens = [token.lower() for token in re.split(r"[_\-.]", name) if token]
    found = {SPLIT_TOKENS[token] for token in tokens if token in SPLIT_TOKENS}
    return found.pop() if len(found) == 1 else None


def detect_layout(release: Path) -> dict:
    """Find each official split's Kaldi metadata directory and WAV directories."""
    metadata, wavs, ignored = defaultdict(list), defaultdict(set), []
    for directory, _subdirs, files in os.walk(release):
        directory = Path(directory)
        relative = directory.relative_to(release).as_posix()
        if {"text", "utt2spk"}.issubset(files):
            split = _split_of(directory.name)
            if split is None:
                ignored.append(relative)  # e.g. an all-splits directory; recorded, not used
            else:
                metadata[split].append(relative)
        if any(name.lower().endswith(".wav") for name in files):
            parts = Path(relative).parts
            split = next((s for s in (_split_of(part) for part in reversed(parts)) if s), None)
            if split is None:
                raise DataError(f"cannot map WAV directory to a split: {relative}")
            top = next(part for part in parts if _split_of(part) == split)
            wavs[split].add(Path(*parts[:parts.index(top) + 1]).as_posix())
    layout = {}
    for split in ("train", "dev", "test"):
        segmented = [d for d in metadata[split] if "segmented" in Path(d).name.lower()]
        if len(metadata[split]) > 1 and len(segmented) == 1:
            # The WAVs are per-utterance segments, so the segmented Kaldi directory describes them.
            ignored += [d for d in metadata[split] if d != segmented[0]]
            metadata[split] = segmented
        if len(metadata[split]) != 1:
            raise DataError(f"expected exactly one Kaldi metadata directory for {split}, found {metadata[split]}; "
                            "inspect the archive and pass an explicit --layout file")
        if not wavs[split]:
            raise DataError(f"no WAV directory found for {split}")
        layout[split] = {"metadata_dir": metadata[split][0], "wav_dirs": sorted(wavs[split])}
    if ignored:
        layout["_ignored_metadata_dirs"] = sorted(ignored)
    return layout


# ----------------------------------------------------------------- manifests

def recording_ids(records: dict, speakers: dict, segments: dict | None) -> tuple[dict, str, list]:
    """Return per-utterance recording/session IDs and the method used.

    Kaldi ``segments`` is authoritative. Otherwise the session is derived from
    ``<speaker>-<session>_<start>_<end>_<n>_<m>`` only when the speaker prefix
    agrees with ``utt2spk``; a non-matching ID gets ``null``.
    """
    if segments:
        mapping = {utt: value.split()[0] if value.split() else None for utt, (value, _) in segments.items()}
        if len(set(mapping.values())) < len(mapping):  # segments that point to themselves carry no grouping
            return mapping, "kaldi_segments_recording_id", []
    result, failures = {}, []
    for utt in records:
        speaker = speakers.get(utt, ("", ""))[0].strip()
        match = ID_PATTERN.match(utt[len(speaker) + 1:]) if speaker and utt.startswith(speaker + "-") else None
        result[utt] = match["session"] if match else None
        if not match:
            failures.append(utt)
    return result, "derived_session_from_utterance_id(<speaker>-<session>_<start>_<end>_<n>_<m>)", failures


def build_raw_manifests(raw: Path = RAW, manifests: Path = MANIFESTS, root: Path = ROOT,
                        layout_override: dict | None = None) -> dict:
    """Write contract manifests and verbatim references for all official splits."""
    release = raw / "release"
    if not release.is_dir():
        raise DataError("run `hparl-extract` first")
    layout = layout_override or detect_layout(release)
    write_output(raw / "LAYOUT.json", json_bytes(layout))
    inventory = {"dataset": DATASET, "layout": layout, "splits": {}}
    acquisition = raw / "ACQUISITION.json"
    if acquisition.exists():
        inventory["archive_sha256"] = read_json(acquisition)["archive_sha256"]
    for split in ("train", "dev", "test"):
        place = layout[split]
        meta = release / place["metadata_dir"]
        files = {name: (meta / name).read_bytes() for name in METADATA_FILES if (meta / name).is_file()}
        texts, issues = kaldi_records(files["text"], f"{place['metadata_dir']}/text")
        speakers, speaker_issues = kaldi_records(files["utt2spk"], f"{place['metadata_dir']}/utt2spk")
        issues += speaker_issues
        segments = kaldi_records(files["segments"], "segments")[0] if "segments" in files else None
        recordings, method, derivation_failures = recording_ids(texts, speakers, segments)
        audio_by_id: dict[str, list[Path]] = defaultdict(list)
        for wav_dir in place["wav_dirs"]:
            for path in (release / wav_dir).rglob("*"):
                if path.is_file() and path.suffix.lower() == ".wav":
                    audio_by_id[path.stem].append(path)
        rows, references = [], []
        for utt in sorted(texts):
            value, _line = texts[utt]
            candidates = audio_by_id.get(utt, [])
            if utt not in speakers:
                issues.append({"issue": "missing_utt2spk", "original_id": utt})
            if not candidates:
                issues.append({"issue": "missing_audio", "original_id": utt})
                continue
            if len(candidates) > 1:
                issues.append({"issue": "ambiguous_audio", "original_id": utt,
                               "paths": sorted(relative_posix(p, root) for p in candidates)})
                continue
            try:
                audio = probe_wav(candidates[0])
            except DataError as error:
                issues.append({"issue": "audio_unreadable", "original_id": utt, "detail": str(error),
                               "audio_path": relative_posix(candidates[0], root)})
                continue
            if audio["num_frames"] == 0:
                issues.append({"issue": "zero_frames", "original_id": utt,
                               "audio_path": relative_posix(candidates[0], root)})
                continue
            speaker = speakers.get(utt, ("", ""))[0].strip() or None
            rows.append(contract_row(
                id=f"hparl:{utt}", source="hparl", original_id=utt, source_split=split, split=split,
                audio_path=relative_posix(candidates[0], root), audio_sha256=audio["audio_sha256"],
                sampling_rate=audio["sampling_rate"], channels=audio["channels"],
                num_frames=audio["num_frames"], duration_seconds=audio["duration_seconds"],
                source_recording_id=recordings.get(utt), speaker_id=speaker))
            references.append({"id": f"hparl:{utt}", "reference": value})
        unlabeled = sorted(set(audio_by_id) - set(texts))
        for stem in unlabeled:
            issues.append({"issue": "audio_without_transcript", "original_id": stem,
                           "paths": sorted(relative_posix(p, root) for p in audio_by_id[stem])})
        manifest_sha = write_output(manifests / f"hparl_{split}.jsonl", jsonl_bytes(rows))
        reference_sha = write_output(manifests / "references" / f"hparl_{split}.jsonl", jsonl_bytes(references))
        inventory["splits"][split] = {
            "metadata_files_sha256": {f"{place['metadata_dir']}/{name}": sha256_bytes(data)
                                      for name, data in sorted(files.items())},
            "transcript_lines": len(texts), "manifest_records": len(rows),
            "audio_seconds": sum(row["duration_seconds"] for row in rows),
            "speakers": len({row["speaker_id"] for row in rows if row["speaker_id"]}),
            "recordings_or_sessions": len({row["source_recording_id"] for row in rows if row["source_recording_id"]}),
            "recording_id_method": method, "recording_id_derivation_failures": len(derivation_failures),
            "sampling_rates": dict(Counter(str(row["sampling_rate"]) for row in rows)),
            "channels": dict(Counter(str(row["channels"]) for row in rows)),
            "manifest_sha256": manifest_sha, "references_sha256": reference_sha,
            "issues": sorted(issues, key=lambda item: (item["issue"], item.get("original_id", ""))),
            "issue_counts": dict(sorted(Counter(item["issue"] for item in issues).items())),
        }
    write_output(manifests / "hparl_inventory.json", json_bytes(inventory))
    return inventory
