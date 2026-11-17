"""Compare the shared scorer to JiWER on deterministic, non-benchmark fixtures."""
import argparse
import importlib.metadata
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jiwer
from greek_asr.scoring import normalize_text, score_corpus


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/scorer_oracle.json"))
    args = parser.parse_args()
    if args.cases < 1:
        parser.error("--cases must be positive")
    rng = random.Random(args.seed)
    vocabulary = ["νόμος", "βουλή", "άρθρο", "ΝΑΙ", "12", "αύριο,", "σήμερα;", "άλφα", "ΐ"]
    largest_difference = 0.0
    zero_word_denominators = 0
    for case in range(args.cases):
        references, predictions = {}, {}
        for index in range(rng.randint(1, 4)):
            identity = str(index)
            references[identity] = "  ".join(rng.choices(vocabulary, k=rng.randint(0, 9)))
            predictions[identity] = "\t".join(rng.choices(vocabulary, k=rng.randint(0, 11)))
        actual = score_corpus(references, predictions)
        ref = [normalize_text(references[k]) for k in sorted(references)]
        hyp = [normalize_text(predictions[k]) for k in sorted(references)]
        for name, oracle, rate in (("word_counts", jiwer.process_words(ref, hyp), "wer"),
                                  ("character_counts", jiwer.process_characters(ref, hyp), "cer")):
            counts = actual[name]
            assert counts["errors"] == oracle.substitutions + oracle.insertions + oracle.deletions, (case, name)
            assert counts["reference_units"] == oracle.hits + oracle.substitutions + oracle.deletions
            if counts["reference_units"]:
                difference = abs(actual[rate] - getattr(oracle, rate))
                assert difference < 1e-12, (case, name, difference)
                largest_difference = max(largest_difference, difference)
            else:
                assert actual[rate] is None  # Explicit policy: undefined, rather than a finite rate.
        zero_word_denominators += int(actual["word_counts"]["reference_units"] == 0)
    report = {"schema": "scorer-oracle-check-v1", "fixture_corpora": args.cases,
              "seed": args.seed, "jiwer": importlib.metadata.version("jiwer"),
              "rapidfuzz": importlib.metadata.version("rapidfuzz"),
              "maximum_nonzero_denominator_rate_difference": largest_difference,
              "zero_word_denominator_cases": zero_word_denominators,
              "zero_denominator_policy": "our rate is null; insertion counts still checked",
              "alignment_note": "tied alignments may allocate S/I/D differently; total distance is checked",
              "real_model_predictions": False, "passed": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
