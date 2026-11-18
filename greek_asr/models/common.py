"""Shared vocabulary, manifest/reference loading, audio decoding, grouped
subset selection, split statistics and prediction I/O for Adnan's attention,
RNN-T and Whisper fine-tuning code.

Audio-side validation (hash, WAV header, ID format, split/source rules) is
not reimplemented here: it reuses ``greek_asr.infer.load_manifest`` so this
module only adds what the inference adapters do not need -- the reference
join, in-memory resampling, batching helpers, grouped subset selection and
the prediction JSONL writer.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

from greek_asr.infer import load_manifest as _validate_audio_manifest

if TYPE_CHECKING:
    import torch

PAD, SOS, EOS, UNK, BLANK = "<pad>", "<sos>", "<eos>", "<unk>", "<blank>"


# --------------------------------------------------------------------------- #
# Manifest + reference loading (train/dev only; never the frozen test split).
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Utterance:
    """One validated training/dev item: audio path plus its joined reference."""

    id: str
    audio_path: Path
    duration_seconds: float
    reference: str


def _load_reference_jsonl(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if (not isinstance(row, dict) or not isinstance(row.get("id"), str)
                or not row["id"] or not isinstance(row.get("reference"), str)):
            raise ValueError(f"{path}:{line_number}: each row needs a nonempty string id and reference")
        if row["id"] in rows:
            raise ValueError(f"{path}:{line_number}: duplicate reference id {row['id']!r}")
        rows[row["id"]] = row["reference"]
    if not rows:
        raise ValueError(f"{path}: empty reference file")
    return rows


def load_utterances(manifest_path: Path, references_path: Path, *,
                    project_root: Path, split: str) -> list[Utterance]:
    """Join manifest and references by ID. Never silently drops a missing/extra ID."""
    if split not in ("train", "dev"):
        raise ValueError("sequence-model training code never reads the frozen test split directly")
    audio_rows = _validate_audio_manifest(manifest_path, project_root, split)
    references = _load_reference_jsonl(references_path)
    manifest_ids = {row["id"] for row in audio_rows}
    reference_ids = set(references)
    if manifest_ids != reference_ids:
        missing = sorted(manifest_ids - reference_ids)
        unexpected = sorted(reference_ids - manifest_ids)
        raise ValueError(f"manifest/reference ID mismatch: missing={missing!r} unexpected={unexpected!r}")
    return [Utterance(id=row["id"], audio_path=row["audio"], duration_seconds=row["duration_seconds"],
                      reference=references[row["id"]]) for row in audio_rows]


def load_waveform(path: Path, *, target_sample_rate: int) -> "torch.Tensor":
    """Decode original PCM16 WAV bytes and resample only in memory; the source file is never rewritten.

    Reads via the stdlib ``wave`` module (matching greek_asr.pretrained.whisper's
    ``_read_pcm16``) instead of ``torchaudio.load``, which requires the separate
    ``torchcodec`` package in this torchaudio version.
    """
    import wave

    import torch
    import torchaudio

    with wave.open(str(path), "rb") as handle:
        if handle.getcomptype() != "NONE":
            raise ValueError(f"{path}: expected uncompressed PCM WAV")
        if handle.getnchannels() != 1:
            raise ValueError(f"{path}: expected mono audio, found {handle.getnchannels()} channels")
        sample_rate, frames = handle.getframerate(), handle.getnframes()
        pcm = handle.readframes(frames)
    if len(pcm) != frames * handle.getsampwidth():
        raise ValueError(f"{path}: truncated WAV payload")
    waveform = torch.frombuffer(bytearray(pcm), dtype=torch.int16).to(torch.float32) / 32768.0
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
    return waveform


# --------------------------------------------------------------------------- #
# Character-level vocabulary (learned only from train references).
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Vocabulary:
    stoi: dict[str, int]
    itos: list[str]

    @property
    def size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        unk = self.stoi.get(UNK)
        ids = []
        for character in text:
            if character in self.stoi:
                ids.append(self.stoi[character])
            elif unk is not None:
                ids.append(unk)
            else:
                raise ValueError(f"character {character!r} is out of vocabulary and no {UNK} token exists")
        return ids

    def decode(self, ids: Sequence[int], *, stop_tokens: Sequence[str] = ()) -> str:
        characters = []
        specials = {PAD, SOS, EOS, BLANK}
        for identifier in ids:
            token = self.itos[identifier]
            if token in stop_tokens:
                break
            if token not in specials:
                characters.append(token)
        return "".join(characters)

    def to_json(self) -> dict:
        return {"schema": "greek-asr-char-vocab-v1", "itos": self.itos}

    @staticmethod
    def from_json(data: Mapping) -> "Vocabulary":
        itos = list(data["itos"])
        return Vocabulary(stoi={token: index for index, token in enumerate(itos)}, itos=itos)


def build_char_vocabulary(references: Sequence[str], specials: Sequence[str]) -> Vocabulary:
    """Character-level vocabulary learned only from the given (train-only) references.

    Never fit this on dev/test text; callers must pass train references only.
    """
    characters = sorted({character for text in references for character in text} - set(specials))
    itos = list(specials) + characters
    if len(itos) != len(set(itos)):
        raise ValueError("special tokens collide with a corpus character")
    return Vocabulary(stoi={token: index for index, token in enumerate(itos)}, itos=itos)


# --------------------------------------------------------------------------- #
# Shared log-mel frontend and padding.
# --------------------------------------------------------------------------- #


class LogMelFrontend:
    """80-dim log-mel filterbank frontend shared by the attention and RNN-T encoders."""

    def __init__(self, sample_rate: int = 16000, n_mels: int = 80, n_fft: int = 400, hop_length: int = 160):
        import torchaudio

        self.sample_rate, self.n_mels, self.hop_length = sample_rate, n_mels, hop_length
        self._mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

    def __call__(self, waveform: "torch.Tensor") -> "torch.Tensor":
        import torch

        mel = self._mel(waveform.unsqueeze(0) if waveform.dim() == 1 else waveform)
        log_mel = torch.log(mel.clamp_min(1e-10))
        return log_mel.squeeze(0).transpose(0, 1)  # (time, n_mels)

    def output_length(self, num_samples: int) -> int:
        return num_samples // self.hop_length + 1


def pad_sequences(sequences: Sequence["torch.Tensor"]) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Zero-pad a list of (time, ...) tensors to the batch's longest length; returns true lengths too."""
    import torch

    if not sequences:
        raise ValueError("cannot pad an empty batch")
    lengths = torch.tensor([sequence.shape[0] for sequence in sequences], dtype=torch.long)
    max_length = int(lengths.max())
    padded = torch.zeros(len(sequences), max_length, *sequences[0].shape[1:], dtype=sequences[0].dtype)
    for index, sequence in enumerate(sequences):
        padded[index, :sequence.shape[0]] = sequence
    return padded, lengths


def padding_mask(lengths: "torch.Tensor", max_length: int) -> "torch.Tensor":
    """True at real (non-padding) positions, shape (batch, max_length)."""
    import torch

    positions = torch.arange(max_length)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)


# --------------------------------------------------------------------------- #
# Prediction JSONL writer.
# --------------------------------------------------------------------------- #


def write_predictions(output_path: Path, ids: Sequence[str], texts: Mapping[str, str]) -> None:
    """Write exactly one {"id","text"} line per requested ID; a genuine empty string is kept, never dropped."""
    if set(ids) != set(texts):
        missing = sorted(set(ids) - set(texts))
        unexpected = sorted(set(texts) - set(ids))
        raise ValueError(f"predictions must cover exactly the requested IDs: missing={missing!r} unexpected={unexpected!r}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        for identity in ids:
            text = texts[identity]
            if not isinstance(text, str):
                raise ValueError(f"{identity}: prediction text must be a string, including genuinely empty output")
            stream.write(json.dumps({"id": identity, "text": text}, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# Grouped, seeded subset selection (speaker/recording groups never split).
# --------------------------------------------------------------------------- #


def _group_key(row: Mapping, priority: Sequence[str]) -> tuple[str, str]:
    for field in priority:
        value = row.get(field)
        if value:
            return (field, str(value))
    # No speaker/recording identity available: the utterance itself is the
    # closest grouping unit. Callers must report how often this happens.
    return ("original_id", str(row["original_id"]))


def select_grouped_subset(rows: Sequence[Mapping], *, group_key_priority: Sequence[str],
                          seed: int, target_hours: float | None = None,
                          target_count: int | None = None) -> tuple[list[dict], dict]:
    """Group rows by the first available key in ``group_key_priority``, then assign
    whole groups in a fixed-seed shuffled order until the target is met. A group
    (e.g. one speaker's or one recording's clips) is never split across subsets.
    """
    if (target_hours is None) == (target_count is None):
        raise ValueError("specify exactly one of target_hours or target_count")
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault(_group_key(row, group_key_priority), []).append(dict(row))
    ordered_keys = sorted(groups)
    random.Random(seed).shuffle(ordered_keys)
    selected: list[dict] = []
    total_hours = 0.0
    selected_groups = 0
    for key in ordered_keys:
        if target_count is not None and len(selected) >= target_count:
            break
        if target_hours is not None and total_hours >= target_hours:
            break
        members = groups[key]
        selected.extend(members)
        total_hours += sum(member["duration_seconds"] for member in members) / 3600.0
        selected_groups += 1
    diagnostics = {
        "seed": seed, "groups_total": len(groups), "groups_selected": selected_groups,
        "utterances_selected": len(selected), "hours_selected": total_hours,
        "groups_without_speaker_or_recording_id": sum(1 for key in groups if key[0] == "original_id"),
    }
    return selected, diagnostics


# --------------------------------------------------------------------------- #
# Read-only split statistics (never used to filter/select from test).
# --------------------------------------------------------------------------- #


def split_statistics(rows: Sequence[Mapping], *, speaker_key: str = "speaker_id") -> dict:
    """Speaker/utterance/hour counts for one split. A read-only report, not a selection step."""
    known_speakers = {row[speaker_key] for row in rows if row.get(speaker_key)}
    unknown = sum(1 for row in rows if not row.get(speaker_key))
    return {
        "utterances": len(rows), "known_speakers": len(known_speakers),
        "utterances_with_unknown_speaker": unknown,
        "total_hours": sum(row["duration_seconds"] for row in rows) / 3600.0,
    }


def literal_overlap(train_references: Sequence[str], test_references: Sequence[str]) -> dict:
    """Exact (unnormalized) string-equality overlap between a training condition and the
    frozen test references. Read-only text comparison; never runs a model on test audio.
    """
    train_set = set(train_references)
    matches = [text for text in test_references if text in train_set]
    return {
        "test_examples": len(test_references),
        "literal_matches": len(matches),
        "unique_matched_texts": len(set(matches)),
    }


# --------------------------------------------------------------------------- #
# Checkpoint save/load (model + optimizer + scheduler + step + RNG + vocab).
# --------------------------------------------------------------------------- #


def capture_rng_state() -> dict:
    import torch

    return {"torch": torch.get_rng_state(), "python": random.getstate()}


def restore_rng_state(state: Mapping) -> None:
    import torch

    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])


def save_checkpoint(path: Path, *, model, optimizer, scheduler, step: int,
                    config: Mapping, vocab: Vocabulary) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": "greek-asr-sequence-checkpoint-v1",
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "step": step,
        "config": dict(config),
        "vocab": vocab.to_json(),
        "rng_state": capture_rng_state(),
    }, path)


def load_checkpoint(path: Path) -> dict:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "greek-asr-sequence-checkpoint-v1":
        raise ValueError(f"{path}: unrecognized checkpoint schema")
    return checkpoint
