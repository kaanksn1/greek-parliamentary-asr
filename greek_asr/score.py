"""Score explicit reference/prediction JSONL files with the shared WER/CER rules."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .scoring import NORMALIZER, load_jsonl, score_corpus

SCORER_CODE_SHA256 = hashlib.sha256(
    Path(__file__).read_bytes() + Path(__file__).with_name("scoring.py").read_bytes()
).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--allow-test", action="store_true",
                        help="Explicitly permit scoring the reserved test split.")
    args = parser.parse_args(argv)
    if args.split == "test" and not args.allow_test:
        parser.error("--split test requires --allow-test after the experiment protocol is fixed")
    try:
        if args.output.resolve() in (args.references.resolve(), args.predictions.resolve()):
            raise ValueError("output must differ from both input files")
        references, reference_sha = load_jsonl(args.references, "reference")
        predictions, prediction_sha = load_jsonl(args.predictions, "text")
        report = {
            "schema": "greek-asr-score-v1",
            "scorer_code_sha256": SCORER_CODE_SHA256,
            "split": args.split,
            "normalizer": NORMALIZER,
            "inputs": {
                "references": {"path": str(args.references), "sha256": reference_sha},
                "predictions": {"path": str(args.predictions), "sha256": prediction_sha},
            },
            "seed": None,
            **score_corpus(references, predictions),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    except (OSError, ValueError, UnicodeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
