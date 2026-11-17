"""Fetch a tiny deterministic HParl training pilot using bounded public ZIP ranges.

No development/test transcript or audio is read. Original WAV bytes and source
transcript lines are retained; this acquisition step performs no cleaning.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import urllib.request
import wave
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
URL = "https://inventory.clarin.gr/storage/"
CATALOG = "https://inventory.clarin.gr/catalogue_backend/api/registry/metadatarecord/1602/"
TOTAL = 11887118613
LIMIT = 64 * 1024 * 1024
PREFIX = "hparl/hparl_segmented_train/"
HEADERS = {
    "ELG-RESOURCE-ID": "1602", "ELG-RESOURCE-DISTRIBUTION-ID": "1024",
    "filename": "1b1c180978d84429b8f0df18895d946c/2e192bae4738437a8e22519f9fc25817/hparl.zip",
    "scope": "private", "Accept-Encoding": "identity",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class RangeReader(io.RawIOBase):
    """Seekable public HTTP source with a hard cumulative transfer ceiling."""
    def __init__(self):
        self.position = self.downloaded = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError("invalid seek origin")
        position = offset if whence == 0 else self.position + offset if whence == 1 else TOTAL + offset
        if not 0 <= position <= TOTAL:
            raise ValueError("invalid archive seek")
        self.position = position
        return position

    def read(self, size=-1):
        size = TOTAL - self.position if size < 0 else min(size, TOTAL - self.position)
        if not size:
            return b""
        if size > LIMIT or self.downloaded + size > LIMIT:
            raise ValueError("pilot acquisition exceeds the 64 MiB network limit")
        end = self.position + size - 1
        request = urllib.request.Request(URL, headers={**HEADERS, "Range": f"bytes={self.position}-{end}"})
        with urllib.request.urlopen(request, timeout=45) as response:
            expected = f"bytes {self.position}-{end}/{TOTAL}"
            if response.status != 206 or response.headers.get("Content-Range") != expected:
                raise ValueError("server did not honor the exact bounded byte range")
            data = response.read(size + 1)
        if len(data) != size:
            raise ValueError("truncated or oversized archive range")
        self.position += size
        self.downloaded += size
        return data


def original_records(data):
    records = {}
    for line in data.decode("utf-8").splitlines(keepends=True):
        fields = line.rstrip("\r\n").split(maxsplit=1)
        if not fields:
            continue
        if len(fields) != 2 or fields[0] in records:
            raise ValueError("invalid or duplicate training metadata ID")
        records[fields[0]] = (fields[1], line)
    return records


def keep_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"refusing to replace different existing bytes: {path}")
    else:
        with path.open("xb") as handle:
            handle.write(data)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "data/raw/pilot")
    args = parser.parse_args()
    if not 1 <= args.count <= 100:
        parser.error("--count must be between 1 and 100; transfer remains capped at 64 MiB")
    output = args.output.resolve()
    output.relative_to(ROOT)
    with urllib.request.urlopen(CATALOG, timeout=30) as response:
        catalog_bytes = response.read(1024 * 1024 + 1)
    if len(catalog_bytes) > 1024 * 1024:
        raise ValueError("unexpectedly large catalog record")
    json.loads(catalog_bytes)
    reader = RangeReader()
    manifest, source_lines = [], []
    with zipfile.ZipFile(reader) as archive:
        members = archive.infolist()
        central_directory = [{"name": x.filename, "bytes": x.file_size, "crc32": f"{x.CRC:08x}"} for x in members]

        def read_member(name):
            info = archive.getinfo(name)
            if info.file_size > LIMIT:
                raise ValueError("oversized ZIP entry")
            return archive.read(info)  # zipfile checks the entry CRC32.

        text_bytes = read_member(PREFIX + "text")
        speaker_bytes = read_member(PREFIX + "utt2spk")
        text, speakers = original_records(text_bytes), original_records(speaker_bytes)
        train_wavs = sorted((x for x in members if not x.is_dir() and x.filename.startswith("hparl/train_wavs/")
                             and x.filename.lower().endswith(".wav")), key=lambda x: x.filename)
        chosen = sorted(random.Random(args.seed).sample(train_wavs, args.count), key=lambda x: x.filename)
        for number, info in enumerate(chosen, 1):
            source_id = PurePosixPath(info.filename).stem
            reference, text_line = text[source_id]
            speaker, speaker_line = speakers[source_id]
            data = read_member(info.filename)
            with wave.open(io.BytesIO(data), "rb") as audio:
                properties = {"sampling_rate": audio.getframerate(), "channels": audio.getnchannels(),
                              "sample_width_bytes": audio.getsampwidth(), "frames": audio.getnframes(),
                              "compression_type": audio.getcomptype(),
                              "duration_seconds": audio.getnframes() / audio.getframerate()}
            pilot_id = f"train-pilot-{number:04d}"
            filename = f"audio/{pilot_id}.wav"
            keep_bytes(output / filename, data)
            manifest.append({"id": pilot_id, "reference": reference, "source_split": "train",
                             "source_utterance_id": source_id, "source_member": info.filename,
                             "source_speaker_id": speaker, "audio_path": filename, "audio_bytes": len(data),
                             "audio_sha256": sha256(data), "source_crc32": f"{info.CRC:08x}", **properties})
            source_lines.append({"source_utterance_id": source_id, "text_member": PREFIX + "text",
                                 "utt2spk_member": PREFIX + "utt2spk", "original_text_line": text_line,
                                 "original_utt2spk_line": speaker_line})
    keep_bytes(output / "manifest.jsonl", b"".join((json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8") for x in manifest))
    keep_bytes(output / "selected_source_lines.json", json_bytes(source_lines))
    summary = {"source_catalog": CATALOG, "catalog_sha256": sha256(catalog_bytes), "archive_url": URL,
               "archive_bytes": TOTAL, "archive_member": HEADERS["filename"],
               "zip_directory_sha256": sha256(json_bytes(central_directory)),
               "training_metadata_sha256": {PREFIX + "text": sha256(text_bytes), PREFIX + "utt2spk": sha256(speaker_bytes)},
               "count": args.count, "seed": args.seed, "selection": "seeded sample from sorted official training WAV members",
               "archive_transfer_bytes": reader.downloaded, "archive_transfer_limit_bytes": LIMIT,
               "reference_policy": "original source training transcript, no text normalization or correction",
               "official_test_data_read": False, "official_development_data_read": False,
               "total_audio_seconds": sum(row["duration_seconds"] for row in manifest)}
    keep_bytes(output / "source_summary.json", json_bytes(summary))
    print(json.dumps({"output": str(output), "files": len(manifest), "audio_seconds": summary["total_audio_seconds"],
                      "archive_transfer_bytes": reader.downloaded}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
