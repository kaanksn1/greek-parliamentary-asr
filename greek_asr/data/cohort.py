"""Split/cohort facts computed only from committed manifests and references.

Nothing here reads audio or changes data, so it also runs on a clean clone.

* Literal test-sentence overlap: a frozen test reference string that is equal,
  with no transformation at all, to at least one target string of a training
  condition. Reported as matched test examples and as distinct matched texts.
* Normalized distinct-text overlap: annotation tags removed, Unicode NFC,
  casefold and whitespace collapse on both sides, then distinct texts are
  intersected. It is reported under that name, never as literal equality.
* Speakers, utterances, hours and sessions per split and per training
  condition, pairwise speaker/session overlaps, and test examples whose
  speaker also occurs in training. Unknown IDs are counted, never invented
  and never silently removed from a speaker count.
"""
from __future__ import annotations

import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable

from .common import ROOT, read_jsonl, sha256_bytes, sha256_file, strip_annotation_tags

TEST = ("data/manifests/test.jsonl", "data/manifests/references/test.jsonl")
TRAINING_CONDITIONS = {
    "raw_hparl_train": ("data/manifests/raw/hparl_train.jsonl", "data/manifests/raw/references/hparl_train.jsonl"),
    "clean_hparl_only": ("data/manifests/train.jsonl", "data/manifests/references/train.jsonl"),
    "clean_hparl_css10": ("data/manifests/train_hparl_css10.jsonl",
                          "data/manifests/references/train_hparl_css10.jsonl"),
}
RAW_SPLITS = {"train": "data/manifests/raw/hparl_train.jsonl", "dev": "data/manifests/raw/hparl_dev.jsonl",
              "test": "data/manifests/test.jsonl"}
NORMALIZATION = "remove <...> annotation tags, Unicode NFC, casefold, collapse whitespace; distinct texts"
CSS10_PRODUCER_NOTE = ("CSS10 Greek: üretici tek okuyucu (Rapunzelina) ve tek kitap (Penelope Delta, "
                       "'Paramythi horis onoma') bildiriyor (kaynak: docs/SECONDARY_SOURCE_TR.md, Kyubyong/css10); "
                       "örnek bazında konuşmacı veya kayıt kimliği verilmediği için speaker_id ve "
                       "source_recording_id boş bırakıldı ve bilinmeyen olarak sayıldı.")


def normalized_key(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", strip_annotation_tags(text)).casefold().split())


def literal_overlap(train_texts: Iterable[str], test_references: list[tuple[str, str]]) -> dict:
    """Exact string equality between test references and training targets."""
    train = set(train_texts)
    matched = sorted((identity, text) for identity, text in test_references if text in train)
    return {"matched_test_examples": len(matched),
            "matched_distinct_test_texts": len({text for _, text in matched}),
            "matched_test_ids_sha256": sha256_bytes("\n".join(i for i, _ in matched).encode("utf-8"))}


def normalized_overlap(train_texts: Iterable[str], test_references: list[tuple[str, str]]) -> dict:
    train = {normalized_key(text) for text in train_texts} - {""}
    keys = [(identity, normalized_key(text)) for identity, text in test_references]
    matched = [(identity, key) for identity, key in keys if key and key in train]
    return {"transformations": NORMALIZATION, "matched_test_examples": len(matched),
            "matched_distinct_normalized_texts": len({key for _, key in matched})}


def cohort_stats(rows: list[dict]) -> dict:
    speakers = {r["speaker_id"] for r in rows if r.get("speaker_id")}
    sessions = {r["source_recording_id"] for r in rows if r.get("source_recording_id")}
    return {"records": len(rows), "hours": round(sum(r["duration_seconds"] for r in rows) / 3600, 3),
            "records_by_source": dict(sorted(Counter(r["source"] for r in rows).items())),
            "known_speakers": len(speakers),
            "records_without_speaker_id": sum(1 for r in rows if not r.get("speaker_id")),
            "known_sessions_or_recordings": len(sessions),
            "records_without_session_id": sum(1 for r in rows if not r.get("source_recording_id"))}


def _file_identity(root: Path, relative: str) -> dict:
    return {"path": relative, "sha256": sha256_file(root / relative)}


def cohort_summary(root: Path = ROOT) -> dict:
    root = Path(root)
    test_rows = read_jsonl(root / TEST[0])
    test_refs = [(r["id"], r["reference"]) for r in read_jsonl(root / TEST[1])]
    test_speakers = {r["id"]: r.get("speaker_id") for r in test_rows}
    raw = {split: read_jsonl(root / path) for split, path in RAW_SPLITS.items() if (root / path).exists()}
    splits = {split: cohort_stats(rows) for split, rows in raw.items()}
    if (root / "data/manifests/dev.jsonl").exists():
        splits["clean_dev"] = cohort_stats(read_jsonl(root / "data/manifests/dev.jsonl"))
    overlaps = {}
    names = [s for s in ("train", "dev", "test") if s in raw]
    for position, first in enumerate(names):
        for second in names[position + 1:]:
            a, b = raw[first], raw[second]
            overlaps[f"{first}&{second}"] = {
                "speakers": len({r["speaker_id"] for r in a if r.get("speaker_id")} &
                                {r["speaker_id"] for r in b if r.get("speaker_id")}),
                "sessions_or_recordings": len({r["source_recording_id"] for r in a if r.get("source_recording_id")} &
                                              {r["source_recording_id"] for r in b if r.get("source_recording_id")})}
    conditions = {}
    for name, (manifest, references) in TRAINING_CONDITIONS.items():
        if not (root / manifest).exists() or not (root / references).exists():
            continue
        rows = read_jsonl(root / manifest)
        texts = [r["reference"] for r in read_jsonl(root / references)]
        speakers = {r["speaker_id"] for r in rows if r.get("speaker_id")}
        conditions[name] = {
            "manifest": _file_identity(root, manifest), "references": _file_identity(root, references),
            **cohort_stats(rows),
            "test_examples_whose_speaker_is_in_this_training": sum(1 for s in test_speakers.values()
                                                                   if s and s in speakers),
            "test_speakers_also_in_this_training": len({s for s in test_speakers.values() if s} & speakers),
            "literal_test_overlap": literal_overlap(texts, test_refs),
            "normalized_test_overlap": normalized_overlap(texts, test_refs),
        }
    return {"schema": "p3-cohort-summary-v1",
            "test": {"manifest": _file_identity(root, TEST[0]), "references": _file_identity(root, TEST[1]),
                     "records": len(test_rows), "distinct_reference_texts": len({t for _, t in test_refs})},
            "split_policy": ("official HParl 1.0.0 session-based train/dev/test split; it is not speaker-disjoint"),
            "splits": splits, "pairwise_overlaps": overlaps, "training_conditions": conditions,
            "css10_identity_note": CSS10_PRODUCER_NOTE}
