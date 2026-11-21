import unittest

from greek_asr.evaluation import normalize_asr, score


class GreekPolicyTests(unittest.TestCase):
    def test_unicode_case_punctuation_and_tags(self):
        self.assertEqual(normalize_asr('  <spoken_noise> ΚΑΛΗΜΈΡΑ, ΚΌΣΜΕ!  '), 'καλημέρα κόσμε')
        self.assertEqual(normalize_asr('λόγος—λόγος'), 'λόγοσ λόγοσ')
        self.assertEqual(normalize_asr('ά\u03b9  α\u0301ι'), 'άι άι')

    def test_accents_digits_and_symbols_are_retained(self):
        self.assertNotEqual(normalize_asr('ή'), normalize_asr('η'))
        self.assertEqual(normalize_asr('3 €'), '3 €')
        self.assertNotEqual(normalize_asr('3'), normalize_asr('τρία'))

    def test_idempotent(self):
        for text in ('<noise>  ΝΌΜΟΣ; 23', 'απ’ αυτό', '', ' €  + '):
            self.assertEqual(normalize_asr(normalize_asr(text)), normalize_asr(text))


class CorpusScoreTests(unittest.TestCase):
    def setUp(self):
        try:
            import jiwer
        except ImportError:
            self.skipTest('requires pinned requirements-verification.txt')

    def test_corpus_weighting_and_known_errors(self):
        result, rows = score({'a': 'ένα δύο τρία', 'b': 'τέσσερα'}, {'a': 'ένα δύο', 'b': 'λάθος'})
        self.assertEqual(result['word_counts']['deletions'], 1)
        self.assertEqual(result['word_counts']['substitutions'], 1)
        self.assertEqual(result['word_counts']['reference_units'], 4)
        self.assertEqual(result['wer'], .5)

    def test_empty_references_are_kept_and_insertions_count(self):
        result, rows = score({'a': '<spoken_noise>', 'b': 'ναι'}, {'a': 'ένα δύο', 'b': ''})
        self.assertEqual(result['examples'], 2)
        self.assertEqual(result['wer'], 3.)
        self.assertEqual(result['empty_normalized_references'], 1)

    def test_empty_corpus_denominator_is_null(self):
        result, rows = score({'a': ''}, {'a': 'ναι'})
        self.assertIsNone(result['wer'])
        self.assertEqual(result['word_counts']['insertions'], 1)

    def test_missing_or_extra_prediction_is_rejected(self):
        with self.assertRaises(ValueError):
            score({'a': 'ναι'}, {})
        with self.assertRaises(ValueError):
            score({'a': 'ναι'}, {'a': 'ναι', 'b': 'όχι'})

    def test_character_spaces_are_counted(self):
        result, rows = score({'a': 'α β'}, {'a': 'αβ'})
        self.assertEqual(result['character_counts']['reference_units'], 3)
        self.assertAlmostEqual(result['cer'], 1 / 3)


if __name__ == '__main__':
    unittest.main()
