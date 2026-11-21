"""Frozen Greek ASR scoring, strict complete-cohort validation and speaker CIs."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import unicodedata
from pathlib import Path

from .scoring import load_jsonl

POLICY = {
    "name": "greek-asr-nfc-casefold-punctuation-v1", "status": "final",
    "annotation_tokens": "remove angle-bracket non-speech annotation tags in memory",
    "unicode": "NFC, casefold, NFC", "punctuation": "replace Unicode P* characters by spaces",
    "whitespace": "strip/collapse Unicode whitespace", "diacritics": "retain Greek accents and diaeresis",
    "numbers": "retain digits; no number-word conversion", "symbols": "retain non-punctuation symbols",
    "cer_spaces": "include the normalized internal spaces", "alignment": "JiWER 4.0.0 / RapidFuzz",
    "aggregation": "sum S/I/D and reference units across every utterance, then divide",
    "empty_references": "keep all IDs; nonempty hypotheses contribute insertions; zero corpus denominator is null",
    "test_source_files": "never rewritten or filtered",
}


def normalize_asr(text: str) -> str:
    text = re.sub(r"<[^<>\r\n]+>", " ", unicodedata.normalize("NFC", text))
    text = unicodedata.normalize("NFC", text.casefold())
    return " ".join("".join(" " if unicodedata.category(c).startswith("P") else c for c in text).split())


def source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes() + Path(__file__).with_name("scoring.py").read_bytes()).hexdigest()


def score(references: dict[str, str], predictions: dict[str, str]):
    import jiwer

    if importlib.metadata.version("jiwer") != "4.0.0":
        raise RuntimeError("final scoring requires the pinned jiwer==4.0.0")
    if not references or references.keys() != predictions.keys():
        raise ValueError("predictions must match the complete nonempty reference ID set exactly")
    words = dict(substitutions=0, insertions=0, deletions=0, reference_units=0, hypothesis_units=0)
    chars = dict(words)
    details = []
    for identity in sorted(references):
        reference, hypothesis = normalize_asr(references[identity]), normalize_asr(predictions[identity])
        wr, cr = jiwer.process_words(reference, hypothesis), jiwer.process_characters(reference, hypothesis)
        wc = {"substitutions": wr.substitutions, "insertions": wr.insertions, "deletions": wr.deletions,
              "reference_units": len(reference.split()), "hypothesis_units": len(hypothesis.split())}
        cc = {"substitutions": cr.substitutions, "insertions": cr.insertions, "deletions": cr.deletions,
              "reference_units": len(reference), "hypothesis_units": len(hypothesis)}
        for total, current in ((words, wc), (chars, cc)):
            for name, value in current.items():
                total[name] += value
            current["errors"] = sum(current[k] for k in ("substitutions", "insertions", "deletions"))
        details.append({"id": identity, "reference": references[identity], "prediction": predictions[identity],
                        "normalized_reference": reference, "normalized_prediction": hypothesis,
                        "word_counts": wc, "character_counts": cc})
    for total in (words, chars):
        total["errors"] = sum(total[k] for k in ("substitutions", "insertions", "deletions"))
    result = {"examples": len(details), "wer": words["errors"] / words["reference_units"] if words["reference_units"] else None,
              "cer": chars["errors"] / chars["reference_units"] if chars["reference_units"] else None,
              "word_counts": words, "character_counts": chars,
              "empty_normalized_references": sum(not d["normalized_reference"] for d in details),
              "empty_predictions": sum(not d["normalized_prediction"] for d in details)}
    return result, details


def speaker_intervals(details, manifest, *, seed=42, repetitions=1000):
    import numpy as np

    groups = {}
    for row in manifest:
        groups[row["id"]] = row.get("speaker_id") or row.get("source_recording_id") or row["id"]
    speakers = sorted(set(groups.values()))
    position = {speaker: i for i, speaker in enumerate(speakers)}
    values = np.zeros((len(speakers), 4), dtype=np.float64)
    for row in details:
        w, c = row["word_counts"], row["character_counts"]
        values[position[groups[row["id"]]]] += [w["errors"], w["reference_units"], c["errors"], c["reference_units"]]
    weights = np.random.default_rng(seed).multinomial(len(speakers), np.full(len(speakers), 1 / len(speakers)), size=repetitions)
    samples = weights @ values
    result = {"unit": "speaker (fallback recording/id)", "clusters": len(speakers), "seed": seed,
              "resamples": repetitions, "confidence": 0.95}
    for name, numerator, denominator in (("wer", 0, 1), ("cer", 2, 3)):
        valid = samples[:, denominator] > 0
        result[name] = np.quantile(samples[valid, numerator] / samples[valid, denominator], [.025, .975]).tolist() if valid.any() else None
    return result


def evaluate_files(references_path, predictions_path, manifest_path, output_dir, *, protocol=None, split="dev"):
    root = Path(__file__).resolve().parents[1]
    if split == "test":
        from .infer import _test_protocol
        _test_protocol(protocol, manifest_path, root)
    references, ref_hash = load_jsonl(references_path, "reference")
    predictions, pred_hash = load_jsonl(predictions_path, "text")
    manifest_bytes = manifest_path.read_bytes()
    manifest = [json.loads(line) for line in manifest_bytes.decode("utf-8-sig").splitlines() if line.strip()]
    ids = [row["id"] for row in manifest]
    if len(set(ids)) != len(ids) or set(ids) != set(references):
        raise ValueError("manifest/reference cohort differs")
    if protocol is not None:
        fixed = json.loads(protocol.read_text(encoding="utf-8"))
        if fixed.get("scoring_policy") != POLICY["name"] or fixed.get("scorer_code_sha256") != source_sha256():
            raise ValueError("scoring implementation/policy differs from the frozen protocol")
        if split == "test" and fixed.get("test_references_sha256") != ref_hash:
            raise ValueError("test reference hash differs from frozen protocol")
    result, details = score(references, predictions)
    result.update(schema="greek-asr-final-score-v1", split=split, policy=POLICY, scorer_code_sha256=source_sha256(),
                  reference_sha256=ref_hash, predictions_sha256=pred_hash,
                  manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                  audio_seconds=sum(row["duration_seconds"] for row in manifest),
                  jiwer=importlib.metadata.version("jiwer"), rapidfuzz=importlib.metadata.version("rapidfuzz"))
    if split == "test":
        result["speaker_bootstrap_95_ci"] = speaker_intervals(details, manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "score.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (output_dir / "utterances.jsonl").open("w", encoding="utf-8") as stream:
        for row in details:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--protocol", type=Path)
    args = parser.parse_args()
    result = evaluate_files(args.references, args.predictions, args.manifest, args.output,
                            protocol=args.protocol, split=args.split)
    print(json.dumps({k: result[k] for k in ("examples", "wer", "cer", "empty_predictions")}))


if __name__ == "__main__":
    main()
