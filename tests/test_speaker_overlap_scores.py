"""The seen/unseen speaker breakdown must be reproducible from the saved scores and match README and the JSON."""
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('report_speaker_overlap', ROOT / 'scripts/report_speaker_overlap.py')
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


class SpeakerOverlapScoresTest(unittest.TestCase):
    def setUp(self):
        self.saved = json.loads((ROOT / 'results/speaker_overlap_scores.json').read_text(encoding='utf-8'))

    def test_json_is_reproduced_from_saved_scores(self):
        self.assertEqual(report.compute(ROOT), self.saved)

    def test_subsets_cover_the_whole_frozen_test(self):
        integrity = json.loads((ROOT / 'results/data_integrity.json').read_text(encoding='utf-8'))
        test = integrity['cohorts']['test']
        total = sum(self.saved['subsets'][name]['clips'] for name in self.saved['subsets'])
        self.assertEqual(total, test['utterances'])
        self.assertEqual(self.saved['test_speakers'], test['speakers_with_ids'])
        overlap = integrity['training_to_test_overlap']['hparl_train']
        self.assertEqual(self.saved['subsets']['seen_in_training']['speakers'], overlap['speakers'])
        self.assertEqual(self.saved['subsets']['seen_in_training']['clips'], overlap['test_clips_with_shared_speaker'])

    def test_readme_block_is_current(self):
        text = (ROOT / 'README.md').read_text(encoding='utf-8')
        self.assertEqual(text.count(report.START), 1)
        block = text[text.index(report.START):text.index(report.END) + len(report.END)]
        self.assertEqual(block, report.render_block(self.saved))


if __name__ == '__main__':
    unittest.main()
