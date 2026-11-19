"""CTC helpers shared by Kemal's plain CTC, HuBERT-CTC and wav2vec 2.0 code.

- Greedy CTC decoding over the *valid* frames of each utterance only: padded
  frames beyond an utterance's true output length are never decoded.
- CTC feasibility: a target of U labels with R adjacent repeats needs at least
  U + R output frames (a blank must separate two identical labels). Infeasible
  training items are never silently given zero loss (``zero_infinity`` stays
  off); they are reported by ID and either rejected or excluded explicitly.
- Annotation tag stripping for *dev loss/tuning only*. HParl dev/test
  references contain markers such as ``<spoken_noise>`` that the cleaned
  training targets do not; the final scoring policy belongs to the common
  evaluation (docs/TEAM.md, Ozan) and is not decided here. Source files are
  never rewritten.
"""
from __future__ import annotations

import re
from typing import Iterable, Sequence

ANNOTATION_TAG = re.compile(r"<[^<>\s]+>")


def strip_annotation_tags(text: str) -> str:
    """Remove ``<...>`` annotation markers and collapse whitespace (in memory only)."""
    return " ".join(ANNOTATION_TAG.sub(" ", text).split())


def ctc_required_frames(label_ids: Sequence[int]) -> int:
    """Minimum CTC output frames for a target: one per label plus one blank per adjacent repeat."""
    repeats = sum(1 for previous, current in zip(label_ids, label_ids[1:]) if previous == current)
    return len(label_ids) + repeats


def ctc_collapse(frame_ids: Iterable[int], blank_id: int) -> list[int]:
    """Standard CTC collapse: merge adjacent duplicates first, then drop blanks."""
    output: list[int] = []
    previous = None
    for identifier in frame_ids:
        identifier = int(identifier)
        if identifier != previous and identifier != blank_id:
            output.append(identifier)
        previous = identifier
    return output


def ctc_greedy_decode(log_probs, lengths, blank_id: int) -> list[list[int]]:
    """Best-path decoding of a (batch, time, vocab) tensor, using only each item's valid frames."""
    best = log_probs.argmax(dim=-1).cpu()
    lengths = [int(length) for length in lengths]
    if len(lengths) != best.shape[0]:
        raise ValueError("one output length is required per batch item")
    decoded = []
    for row, length in zip(best, lengths):
        if length < 0 or length > row.shape[0]:
            raise ValueError(f"invalid CTC output length {length} for {row.shape[0]} frames")
        decoded.append(ctc_collapse(row[:length].tolist(), blank_id))
    return decoded


def check_ctc_feasibility(items: Sequence[tuple[str, int, Sequence[int]]], *, policy: str,
                          max_excluded_fraction: float) -> tuple[set[str], list[dict]]:
    """Check (id, output_frames, label_ids) triples; returns (excluded IDs, report rows).

    ``policy`` is ``"error"`` (any infeasible or empty-target item fails the run) or
    ``"exclude_and_report"`` (infeasible items are listed in the run record and left out
    of training; more than ``max_excluded_fraction`` of the items still fails the run).
    """
    if policy not in ("error", "exclude_and_report"):
        raise ValueError("infeasible_policy must be 'error' or 'exclude_and_report'")
    problems = []
    for identity, frames, labels in items:
        if not labels:
            problems.append({"id": identity, "reason": "empty_target", "output_frames": frames, "required_frames": 0})
            continue
        required = ctc_required_frames(labels)
        if frames < required:
            problems.append({"id": identity, "reason": "too_few_output_frames",
                             "output_frames": frames, "required_frames": required})
    if problems and policy == "error":
        raise ValueError(f"{len(problems)} training items cannot be aligned by CTC, e.g. {problems[:3]!r}")
    if items and len(problems) / len(items) > max_excluded_fraction:
        raise ValueError(f"{len(problems)}/{len(items)} infeasible CTC items exceed max_excluded_fraction="
                         f"{max_excluded_fraction}; check the frontend/subsampling instead of dropping data")
    return {row["id"] for row in problems}, problems


def length_bucketed_batches(lengths: Sequence[float], batch_size: int, *, seed: int,
                            bucket_batches: int = 50) -> list[list[int]]:
    """Seeded shuffled batches whose members have similar lengths (less padding).

    Indices are shuffled, cut into buckets of ``bucket_batches * batch_size`` items,
    sorted by length inside each bucket, batched, and the batches shuffled again.
    Every index appears exactly once.
    """
    import random

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    generator = random.Random(seed)
    order = list(range(len(lengths)))
    generator.shuffle(order)
    bucket_size = max(batch_size * bucket_batches, batch_size)
    batches = []
    for start in range(0, len(order), bucket_size):
        bucket = sorted(order[start:start + bucket_size], key=lambda index: lengths[index])
        batches.extend(bucket[offset:offset + batch_size] for offset in range(0, len(bucket), batch_size))
    generator.shuffle(batches)
    return batches


def environment_record() -> dict:
    """Package versions and accelerator identity for run records (never fabricated)."""
    import importlib.metadata
    import platform

    versions = {}
    for name in ("torch", "torchaudio", "transformers", "numpy", "huggingface-hub"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    record = {"python": platform.python_version(), "platform": platform.platform(), "packages": versions}
    try:
        import torch

        record.update(torch_build=torch.__version__, hip=torch.version.hip, cuda=torch.version.cuda,
                      gpu_available=torch.cuda.is_available())
        if torch.cuda.is_available():
            record["gpu"] = torch.cuda.get_device_name(0)
            record["peak_gpu_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
    except ImportError:
        pass
    return record
