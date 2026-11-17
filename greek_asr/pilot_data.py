"""Audit downloaded training audio and transcript lineage without editing them."""
from __future__ import annotations

import hashlib
import io
import json
import math
import re
import wave
import zlib
from pathlib import Path, PurePosixPath


def audit_pilot(pilot_dir: Path) -> tuple[list[dict], dict]:
    """Check the local pilot contract; reference semantics still need review."""
    base = pilot_dir.resolve(strict=True)
    filenames = ("manifest.jsonl", "source_summary.json", "selected_source_lines.json")
    raw = {name: (base / name).read_bytes() for name in filenames}
    records = [json.loads(line) for line in raw["manifest.jsonl"].decode("utf-8-sig").splitlines() if line.strip()]
    summary = json.loads(raw["source_summary.json"])
    sources = json.loads(raw["selected_source_lines.json"])
    if not records or not isinstance(summary, dict) or not isinstance(sources, list):
        raise ValueError("invalid or empty pilot metadata")
    source_by_id = {}
    for source in sources:
        identity = source.get("source_utterance_id") if isinstance(source, dict) else None
        if not isinstance(identity, str) or not identity or identity in source_by_id:
            raise ValueError("invalid or duplicate source metadata ID")
        source_by_id[identity] = source

    seen_ids, seen_sources, annotations, audio_checks = set(), set(), {}, []
    total_seconds = 0.0
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("each manifest row must be an object")
        identity, source_id = row.get("id"), row.get("source_utterance_id")
        if any(not isinstance(value, str) or not value for value in (identity, source_id)):
            raise ValueError("manifest IDs must be nonempty strings")
        if identity in seen_ids or source_id in seen_sources:
            raise ValueError("duplicate pilot or source ID")
        seen_ids.add(identity)
        seen_sources.add(source_id)
        if row.get("source_split") != "train":
            raise ValueError(f"{identity}: only the training split is permitted")
        member = PurePosixPath(row.get("source_member", ""))
        if member.parts[:2] != ("hparl", "train_wavs") or ".." in member.parts or member.stem != source_id:
            raise ValueError(f"{identity}: source member does not match the training source ID")
        source = source_by_id.get(source_id, {})
        for field, value in (("original_text_line", row.get("reference")),
                             ("original_utt2spk_line", row.get("source_speaker_id"))):
            line = source.get(field)
            fields = line.rstrip("\r\n").split(maxsplit=1) if isinstance(line, str) else []
            if len(fields) != 2 or fields != [source_id, value]:
                raise ValueError(f"{identity}: source lineage mismatch in {field}")
        if source.get("text_member") != "hparl/hparl_segmented_train/text" or source.get("utt2spk_member") != "hparl/hparl_segmented_train/utt2spk":
            raise ValueError(f"{identity}: source metadata is not from training")
        relative = row.get("audio_path")
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"{identity}: invalid audio path")
        audio_path = (base / relative).resolve(strict=True)
        if Path(relative).is_absolute() or not audio_path.is_relative_to(base):
            raise ValueError(f"{identity}: audio path is outside the pilot directory")
        data = audio_path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != row.get("audio_sha256") or len(data) != row.get("audio_bytes"):
            raise ValueError(f"{identity}: audio hash or byte count mismatch")
        if f"{zlib.crc32(data):08x}" != row.get("source_crc32"):
            raise ValueError(f"{identity}: source audio CRC32 mismatch")
        with wave.open(io.BytesIO(data), "rb") as audio:
            actual = {"sampling_rate": audio.getframerate(), "channels": audio.getnchannels(),
                      "sample_width_bytes": audio.getsampwidth(), "frames": audio.getnframes(),
                      "compression_type": audio.getcomptype()}
            pcm = audio.readframes(audio.getnframes())
        if any(row.get(key) != value for key, value in actual.items()):
            raise ValueError(f"{identity}: WAV header mismatch")
        if len(pcm) != actual["frames"] * actual["channels"] * actual["sample_width_bytes"]:
            raise ValueError(f"{identity}: truncated WAV frames")
        seconds = actual["frames"] / actual["sampling_rate"]
        reported = row.get("duration_seconds")
        if not isinstance(reported, (float, int)) or not math.isclose(seconds, reported, rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"{identity}: duration mismatch")
        total_seconds += seconds
        tags = sorted(set(re.findall(r"<[^<>\r\n]+>", row["reference"])))
        if tags:
            annotations[identity] = tags
        audio_checks.append({"id": identity, "audio_path": relative, "sha256": digest,
                             "duration_seconds": seconds, **actual})
    if seen_sources != source_by_id.keys():
        raise ValueError("source metadata cohort does not match the manifest")
    if summary.get("count") != len(records) or not math.isclose(total_seconds, summary.get("total_audio_seconds", -1), rel_tol=0, abs_tol=1e-9):
        raise ValueError("pilot summary count or duration mismatch")
    if any(summary.get(key) is not False for key in ("official_test_data_read", "official_development_data_read")):
        raise ValueError("pilot source summary does not confirm train-only acquisition")
    diagnostics = {"schema": "train-pilot-audit-v1", "examples": len(records),
                   "audio_files_checked": len(audio_checks), "total_audio_seconds": total_seconds,
                   "inputs_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in raw.items()},
                   "audio": audio_checks, "source_split": "train",
                   "reference_annotation_tags": annotations,
                   "quality_status": "requires_reference_review",
                   "note": "Byte integrity and source lineage checked; no semantic audio/transcript adjudication or correction."}
    return records, diagnostics
