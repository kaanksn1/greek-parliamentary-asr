"""Hand-calculated edit counts and input/report contract checks."""

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from greek_asr.score import main
from greek_asr.scoring import EditCounts, edit_counts, load_jsonl, normalize_text, score_corpus


class ScoringTests(unittest.TestCase):
    def test_hand_computed_edit_counts_and_ties(self):
        counts = edit_counts("a b c d e f".split(), "a x c e f z".split())
        self.assertEqual(counts, EditCounts(1, 1, 1, 6, 6))
        # Substituting both words ties with deleting/inserting; diagonal wins.
        self.assertEqual(edit_counts(["a", "b"], ["b", "a"]), EditCounts(2, 0, 0, 2, 2))

    def test_corpus_weighting_uses_total_reference_words(self):
        result = score_corpus({"short": "a", "long": "b c d e"},
                              {"long": "b c d e", "short": "x"})
        self.assertAlmostEqual(result["wer"], 1 / 5)
        self.assertEqual(result["word_counts"]["substitutions"], 1)
        self.assertEqual(result["word_counts"]["reference_units"], 5)

    def test_error_rate_can_exceed_one(self):
        result = score_corpus({"a": "a"}, {"a": "b c d"})
        self.assertEqual(result["wer"], 3.0)
        self.assertEqual(result["word_counts"], EditCounts(1, 2, 0, 1, 3).as_dict())

    def test_empty_reference_insertions_count_in_corpus(self):
        result = score_corpus({"empty": " \t", "normal": "hello"},
                              {"empty": "noise", "normal": "hello"})
        self.assertEqual(result["wer"], 1.0)
        self.assertEqual(result["word_counts"]["insertions"], 1)
        self.assertEqual(result["word_counts"]["reference_units"], 1)

    def test_zero_corpus_reference_size_has_null_rates_and_counts(self):
        result = score_corpus({"empty": ""}, {"empty": "a b"})
        self.assertIsNone(result["wer"])
        self.assertIsNone(result["cer"])
        self.assertEqual(result["word_counts"]["insertions"], 2)
        self.assertEqual(result["character_counts"]["insertions"], 3)
        self.assertIsNone(score_corpus({}, {})["wer"])

    def test_greek_unicode_and_significant_characters(self):
        self.assertEqual(normalize_text("  ΆΛΦΑ,\t12;  "), "άλφα, 12;")
        result = score_corpus({"a": "Άλφα ΐ"}, {"a": "Α\u0301ΛΦΑ Ι\u0308\u0301"})
        self.assertEqual(result["wer"], 0.0)
        self.assertEqual(result["cer"], 0.0)
        self.assertNotEqual(normalize_text("α"), normalize_text("ά"))
        self.assertEqual(score_corpus({"a": "α"}, {"a": "ά"})["cer"], 1.0)
        self.assertEqual(score_corpus({"a": "α β"}, {"a": "αβ"})["cer"], 1 / 3)

    def test_prediction_cohort_must_match_exactly(self):
        with self.assertRaisesRegex(ValueError, "missing=.*a.*unexpected=.*b"):
            score_corpus({"a": "hello"}, {"b": "hello"})


class InputAndCliTests(unittest.TestCase):
    def write_rows(self, directory, filename, rows):
        path = directory / filename
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                        encoding="utf-8")
        return path

    def test_duplicate_ids_rejected_in_both_input_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            for field in ("reference", "text"):
                with self.subTest(field=field):
                    path = self.write_rows(Path(tmp), field + ".jsonl",
                                           [{"id": "a", field: "one"}, {"id": "a", field: "two"}])
                    with self.assertRaisesRegex(ValueError, "duplicate id"):
                        load_jsonl(path, field)

    def test_record_types_are_rejected(self):
        invalid = [[], {"id": 1, "reference": "a"}, {"id": " ", "reference": "a"},
                   {"id": "a", "reference": None}, {"id": "a"}]
        with tempfile.TemporaryDirectory() as tmp:
            for row in invalid:
                with self.subTest(row=row):
                    path = self.write_rows(Path(tmp), "invalid.jsonl", [row])
                    with self.assertRaises(ValueError):
                        load_jsonl(path, "reference")

    def test_cli_reports_actual_input_hashes_counts_and_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            references = self.write_rows(directory, "refs.jsonl", [{"id": "a", "reference": "Ναι τώρα"}])
            predictions = self.write_rows(directory, "preds.jsonl", [{"id": "a", "text": "ναι"}])
            output = directory / "result.json"
            self.assertEqual(main(["--references", str(references), "--predictions", str(predictions),
                                   "--output", str(output), "--split", "validation"]), 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["schema"], "greek-asr-score-v1")
            self.assertEqual(report["split"], "validation")
            self.assertEqual(report["normalizer"]["status"], "provisional")
            self.assertIsNone(report["seed"])
            self.assertEqual(report["wer"], 0.5)
            self.assertEqual(report["word_counts"]["deletions"], 1)
            self.assertEqual(report["inputs"]["references"]["sha256"],
                             hashlib.sha256(references.read_bytes()).hexdigest())
            self.assertEqual(report["inputs"]["predictions"]["sha256"],
                             hashlib.sha256(predictions.read_bytes()).hexdigest())

    def test_test_scoring_requires_explicit_opt_in_before_reading_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            references = directory / "refs.jsonl"
            predictions = directory / "preds.jsonl"
            output = directory / "test.json"
            args = ["--references", str(references), "--predictions", str(predictions),
                    "--output", str(output), "--split", "test"]
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as stopped:
                main(args)
            self.assertEqual(stopped.exception.code, 2)
            self.assertIn("--allow-test", stderr.getvalue())
            self.assertFalse(output.exists())
            self.write_rows(directory, "refs.jsonl", [{"id": "a", "reference": "ναι"}])
            self.write_rows(directory, "preds.jsonl", [{"id": "a", "text": "ναι"}])
            self.assertEqual(main(args + ["--allow-test"]), 0)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["wer"], 0.0)


if __name__ == "__main__":
    unittest.main()
