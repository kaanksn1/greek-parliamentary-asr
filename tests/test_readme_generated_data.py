"""The README data/overlap block must be exactly what the committed generator writes from results/data_integrity.json."""
import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('summarize_benchmark', ROOT / 'scripts/summarize_benchmark.py')
summarize = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summarize)


class ReadmeGeneratedDataTest(unittest.TestCase):
    def test_block_matches_generator_output(self):
        with tempfile.TemporaryDirectory() as folder:
            temp = Path(folder)
            (temp / 'results').mkdir()
            shutil.copy(ROOT / 'README.md', temp / 'README.md')
            shutil.copy(ROOT / 'results/data_integrity.json', temp / 'results/data_integrity.json')
            summarize.data_block(temp)
            self.assertEqual((temp / 'README.md').read_text(encoding='utf-8'),
                             (ROOT / 'README.md').read_text(encoding='utf-8'),
                             'README data block is stale: run python scripts/summarize_benchmark.py')

    def test_block_markers_present_once(self):
        text = (ROOT / 'README.md').read_text(encoding='utf-8')
        self.assertEqual(text.count(summarize.DATA_START), 1)
        self.assertEqual(text.count(summarize.DATA_END), 1)


if __name__ == '__main__':
    unittest.main()
