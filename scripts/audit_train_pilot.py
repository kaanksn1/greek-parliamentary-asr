"""Audit the downloaded train-only pilot and write a source-hashed report."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from greek_asr.pilot_data import audit_pilot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=Path("data/raw/pilot"))
    parser.add_argument("--output", type=Path, default=Path("outputs/train_pilot_audit.json"))
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.pilot.resolve()):
        parser.error("audit output must be outside the raw pilot directory")
    try:
        _, report = audit_pilot(args.pilot)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({"examples": report["examples"], "audio_seconds": report["total_audio_seconds"],
                      "reference_annotation_tags": report["reference_annotation_tags"],
                      "quality_status": report["quality_status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
