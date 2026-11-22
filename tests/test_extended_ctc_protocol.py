"""The extended CTC/HuBERT addendum may only lengthen training and must stay locked."""
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('run_extended_ctc', ROOT / 'scripts/run_extended_ctc.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class ExtendedCtcProtocolTest(unittest.TestCase):
    def test_only_training_length_fields_change(self):
        for spec in runner.SYSTEMS.values():
            changes = runner.config_changes(spec)
            self.assertTrue(changes)
            self.assertLessEqual(set(changes), {'max_steps', 'warmup_steps', 'dev_eval_every'})
            self.assertGreater(changes['max_steps']['extended'], changes['max_steps']['original'])

    def test_warmup_stays_ten_percent_of_updates(self):
        for spec in runner.SYSTEMS.values():
            config = json.loads((ROOT / spec['config']).read_text(encoding='utf-8'))
            self.assertEqual(config['warmup_steps'] * 10, config['max_steps'])

    def test_extra_model_field_is_rejected(self):
        spec = dict(runner.SYSTEMS['plain_ctc_extended'])
        original = runner.read
        def patched(path):
            value = original(path)
            if Path(path) == ROOT / spec['config']:
                value = dict(value, learning_rate=0.5)
            return value
        runner.read = patched
        try:
            with self.assertRaises(ValueError):
                runner.config_changes(spec)
        finally:
            runner.read = original

    def test_committed_protocol_matches_current_inputs(self):
        protocol = json.loads(runner.PROTOCOL.read_text(encoding='utf-8'))
        for relative, digest in protocol['locked_code_sha256'].items():
            self.assertEqual(runner.sha(ROOT / relative), digest, relative)
        for name, spec in protocol['systems'].items():
            self.assertEqual(runner.sha(ROOT / spec['config']), spec['config_sha256'], name)
        self.assertEqual(runner.sha(ROOT / runner.TEST_MANIFEST), protocol['test_manifest_sha256'])
        self.assertEqual(runner.sha(ROOT / runner.TEST_REFERENCES), protocol['test_references_sha256'])
        for name in protocol['system_order']:
            original = protocol['systems'][name]['original']
            self.assertTrue((ROOT / 'results/benchmark' / original / 'score.json').exists())

    def test_new_outputs_never_overlap_original_rows(self):
        for name, spec in runner.SYSTEMS.items():
            self.assertTrue(name.endswith('_extended'))
            self.assertIn('_extended', spec['checkpoint'])
            self.assertNotEqual(spec['original'], name)


if __name__ == '__main__':
    unittest.main()
