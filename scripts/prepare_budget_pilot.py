"""Acquire a reproducible TRAIN-only timing cohort; never inspect dev/test.

Reuse the owner's byte-range ZIP reader. Keep original audio bytes and hash
them against Samed's committed inventory. This is a runtime measurement cohort,
not a new train/dev split, cleaning operation or benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
import zipfile
from pathlib import Path

import fetch_train_pilot as source

ROOT = Path(__file__).resolve().parents[1]


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def quantiles(values):
    ordered = sorted(values)
    return {str(p): ordered[round((len(ordered) - 1) * p)] for p in (0, .5, .9, .95, .99, 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-count", type=int, default=128)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    output.relative_to(root)
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
    manifest = root / "data/manifests/train.jsonl"
    references = root / "data/manifests/references/train.jsonl"
    rows = read_rows(manifest)
    if any(row["source_split"] != "train" or row["split"] != "train" for row in rows):
        raise ValueError("timing acquisition accepts official TRAIN only")
    refs = {row["id"]: row for row in read_rows(references)}
    rng = random.Random(args.seed)
    random_rows = rng.sample(sorted(rows, key=lambda row: row["id"]), args.random_count)
    longest = sorted(rows, key=lambda row: (-row["duration_seconds"], row["id"]))[:8]
    most_chars = sorted(rows, key=lambda row: (-len(refs[row["id"]]["reference"]), row["id"]))[:8]
    selected = {row["id"]: row for row in random_rows + longest + most_chars}
    output.mkdir(parents=True, exist_ok=True)
    source.LIMIT = 128 * 1024 * 1024
    reader = source.RangeReader()
    started = time.perf_counter()
    acquired = []
    with zipfile.ZipFile(reader) as archive:
        for number, row in enumerate(sorted(selected.values(), key=lambda row: row["id"]), 1):
            destination = output / "audio" / (row["original_id"] + ".wav")
            if destination.exists():
                data = destination.read_bytes()
            else:
                data = archive.read("hparl/train_wavs/" + destination.name)
            if hashlib.sha256(data).hexdigest() != row["audio_sha256"]:
                raise ValueError(f"source audio hash mismatch: {row['id']}")
            source.keep_bytes(destination, data)
            acquired.append({**row, "original_audio_path": row["audio_path"],
                             "audio_path": destination.relative_to(root).as_posix()})
            if number % 16 == 0:
                print(json.dumps({"acquired": number, "total": len(selected),
                                  "transfer_mib": round(reader.downloaded / 1024**2, 2)}), flush=True)
    for name, content in (("train.jsonl", acquired),
                          ("references.jsonl", [refs[row["id"]] for row in acquired])):
        (output / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in content),
                                  encoding="utf-8")
    report = {
        "schema": "p3-train-budget-cohort-v1", "seed": args.seed,
        "selection": "128 seeded uniform clips + 8 longest + 8 most transcript characters; deduplicated by ID",
        "random_ids": [row["id"] for row in random_rows],
        "longest_ids": [row["id"] for row in longest], "most_chars_ids": [row["id"] for row in most_chars],
        "official_dev_or_test_read": False, "benchmark_result": False,
        "input_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "input_references_sha256": hashlib.sha256(references.read_bytes()).hexdigest(),
        "source_archive_bytes": source.TOTAL, "source_transfer_bytes": reader.downloaded,
        "source_transfer_limit_bytes": source.LIMIT, "all_audio_hashes_verified": True,
        "acquisition_seconds": time.perf_counter() - started,
        "full_clean_train": {"utterances": len(rows), "hours": sum(row["duration_seconds"] for row in rows) / 3600,
                             "duration_seconds_quantiles": quantiles([row["duration_seconds"] for row in rows]),
                             "target_characters_quantiles": quantiles([len(refs[row["id"]]["reference"]) for row in rows])},
        "cohort": {"utterances": len(acquired), "hours": sum(row["duration_seconds"] for row in acquired) / 3600,
                   "duration_seconds_quantiles": quantiles([row["duration_seconds"] for row in acquired]),
                   "target_characters_quantiles": quantiles([len(refs[row["id"]]["reference"]) for row in acquired])},
    }
    write_json(output / "acquisition.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
