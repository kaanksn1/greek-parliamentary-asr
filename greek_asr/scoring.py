"""Deterministic corpus WER/CER with an explicitly provisional normalizer.

Accents, digits and punctuation are retained. CER includes the single spaces
left by whitespace normalization. Alignment ties prefer diagonal, deletion,
then insertion. WER/CER were cross-checked with JiWER 4.0.0 using the
deterministic fixtures in scripts/verify_scoring.py; tied S/I/D alignments
need not be identical between libraries.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

NORMALIZER = {
    "name": "nfc-casefold-whitespace-v1",
    "status": "provisional",
    "unicode": "NFC before and after casefold",
    "whitespace": "strip and collapse Unicode whitespace to one ASCII space",
    "diacritics": "retained",
    "digits": "retained",
    "punctuation": "retained",
    "character_unit": "Unicode code points after normalization",
    "cer_spaces": "included after normalization",
    "alignment_tie_break": ["diagonal", "deletion", "insertion"],
    "independent_metric_library_check": "JiWER 4.0.0; see scripts/verify_scoring.py",
}


def normalize_text(text: str) -> str:
    """Apply canonical Unicode, case and whitespace normalization only."""
    text = unicodedata.normalize("NFC", text)
    text = unicodedata.normalize("NFC", text.casefold())
    return " ".join(text.split())


@dataclass(frozen=True)
class EditCounts:
    substitutions: int = 0
    insertions: int = 0
    deletions: int = 0
    reference_units: int = 0
    hypothesis_units: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.insertions + self.deletions

    @property
    def error_rate(self) -> float | None:
        return self.errors / self.reference_units if self.reference_units else None

    def as_dict(self) -> dict[str, int]:
        return {
            "substitutions": self.substitutions,
            "insertions": self.insertions,
            "deletions": self.deletions,
            "errors": self.errors,
            "reference_units": self.reference_units,
            "hypothesis_units": self.hypothesis_units,
        }

    def __add__(self, other: EditCounts) -> EditCounts:
        return EditCounts(
            self.substitutions + other.substitutions,
            self.insertions + other.insertions,
            self.deletions + other.deletions,
            self.reference_units + other.reference_units,
            self.hypothesis_units + other.hypothesis_units,
        )


def edit_counts(reference: Sequence[str], hypothesis: Sequence[str]) -> EditCounts:
    """Levenshtein alignment in O(reference*hypothesis) time and linear space."""
    # Each cell is (distance, substitutions, insertions, deletions).
    previous = [(j, 0, j, 0) for j in range(len(hypothesis) + 1)]
    for i, actual in enumerate(reference, 1):
        current = [(i, 0, 0, i)]
        for j, predicted in enumerate(hypothesis, 1):
            mismatch = int(actual != predicted)
            distance, substitutions, insertions, deletions = previous[j - 1]
            diagonal = (distance + mismatch, substitutions + mismatch, insertions, deletions)
            distance, substitutions, insertions, deletions = previous[j]
            deletion = (distance + 1, substitutions, insertions, deletions + 1)
            distance, substitutions, insertions, deletions = current[j - 1]
            insertion = (distance + 1, substitutions, insertions + 1, deletions)
            # min returns the first equally good candidate, making ties explicit.
            current.append(min((diagonal, deletion, insertion), key=lambda cell: cell[0]))
        previous = current
    _, substitutions, insertions, deletions = previous[-1]
    return EditCounts(substitutions, insertions, deletions, len(reference), len(hypothesis))


def load_jsonl(path: Path, text_key: str) -> tuple[dict[str, str], str]:
    """Load exact IDs/text and return the SHA-256 of the bytes actually parsed."""
    data = path.read_bytes()
    rows: dict[str, str] = {}
    for line_number, line in enumerate(data.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: each row must be a JSON object")
        identity = row.get("id")
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError(f"{path}:{line_number}: id must be a nonempty string")
        if not isinstance(row.get(text_key), str):
            raise ValueError(f"{path}:{line_number}: {text_key} must be a string")
        if identity in rows:
            raise ValueError(f"{path}:{line_number}: duplicate id {identity!r}")
        rows[identity] = row[text_key]
    return rows, hashlib.sha256(data).hexdigest()


def _validate_texts(rows: Mapping[str, str], name: str) -> None:
    for identity, text in rows.items():
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError(f"{name}: id must be a nonempty string")
        if not isinstance(text, str):
            raise ValueError(f"{name}: text for {identity!r} must be a string")


def score_corpus(references: Mapping[str, str], predictions: Mapping[str, str]) -> dict:
    """Sum edit counts across utterances, then divide by corpus reference size."""
    _validate_texts(references, "references")
    _validate_texts(predictions, "predictions")
    if references.keys() != predictions.keys():
        missing = sorted(references.keys() - predictions.keys())
        unexpected = sorted(predictions.keys() - references.keys())
        raise ValueError(f"prediction ID mismatch: missing={missing!r}; unexpected={unexpected!r}")
    words, characters = EditCounts(), EditCounts()
    for identity in sorted(references):
        reference = normalize_text(references[identity])
        prediction = normalize_text(predictions[identity])
        words += edit_counts(reference.split(), prediction.split())
        characters += edit_counts(reference, prediction)
    return {
        "examples": len(references),
        "wer": words.error_rate,
        "cer": characters.error_rate,
        "word_counts": words.as_dict(),
        "character_counts": characters.as_dict(),
    }
