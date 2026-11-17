"""Export the already-audited three training clips to the shared inference schema.

This is an integration fixture, not Samed's full data acquisition/cleaning work.
It never edits the existing pilot or writes reference text into model inputs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from greek_asr.pilot_data import audit_pilot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/owner_pilot_manifest.jsonl")
    args = parser.parse_args()
    if args.output.resolve().is_relative_to((ROOT / "data").resolve()):
        parser.error("integration fixture must not overwrite the source data directory")
    pilot = ROOT / "data/raw/pilot"
    records, _ = audit_pilot(pilot)
    result = []
    for row in records:
        if row["source_split"] != "train":
            raise ValueError("owner runtime pilot accepts original training records only")
        result.append({
            "id": "hparl:" + row["source_utterance_id"], "source": "hparl",
            "original_id": row["source_utterance_id"], "source_split": "train", "split": "train",
            "audio_path": (pilot / row["audio_path"]).relative_to(ROOT).as_posix(),
            "audio_sha256": row["audio_sha256"], "sampling_rate": row["sampling_rate"],
            "channels": row["channels"], "num_frames": row["frames"],
            "duration_seconds": row["duration_seconds"],
            "source_recording_id": None, "speaker_id": row["source_speaker_id"],
        })
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in result).encode("utf-8")
    if args.output.exists() and args.output.read_bytes() != data:
        parser.error("existing output differs; choose a new --output path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(data)
    print(f"Exported {len(result)} unchanged training-audio inputs to {args.output}; no references.")


if __name__ == "__main__":
    main()
