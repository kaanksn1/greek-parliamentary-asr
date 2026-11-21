import contextlib
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace

from greek_asr.pretrained.nemo import NemoBackend


class NemoBatchTests(unittest.TestCase):
    def test_batch_language_options_empty_text_and_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = [Path(temporary)/str(i) for i in range(3)]
            for path in paths:
                path.write_bytes(b'fixture')
            calls = []
            class Model:
                def transcribe(self, supplied, **options):
                    calls.append((supplied, options))
                    return ['a', '', SimpleNamespace(text='c')]
            backend = NemoBackend('nvidia/canary-1b-v2', 'a'*40, 'cpu', 'fixture', Model(),
                                  SimpleNamespace(inference_mode=contextlib.nullcontext), batch_size=4)
            self.assertEqual(backend.transcribe_batch(paths), ['a', '', 'c'])
            self.assertEqual(calls[0][0], [str(p.resolve()) for p in paths])
            self.assertEqual(calls[0][1], {'batch_size': 4, 'source_lang': 'el', 'target_lang': 'el'})

    def test_bad_batch_is_not_silently_padded(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'audio'
            path.write_bytes(b'fixture')
            backend = NemoBackend('nvidia/parakeet-tdt-0.6b-v3', 'a'*40, 'cpu', 'fixture',
                                  SimpleNamespace(transcribe=lambda *args, **kwargs: []),
                                  SimpleNamespace(inference_mode=contextlib.nullcontext), batch_size=4)
            with self.assertRaises(RuntimeError):
                backend.transcribe_batch([path])


class WhisperBatchTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except ImportError:
            self.skipTest('requires project model environment')

    def test_batch_features_keep_real_count_and_attention_mask(self):
        import torch
        from greek_asr.pretrained.whisper import WhisperBackend
        calls = []
        class Processor:
            def __call__(self, audios, **kwargs):
                calls.append(kwargs)
                return {'input_features': torch.zeros(len(audios), 80, 3000),
                        'attention_mask': torch.ones(len(audios), 3000, dtype=torch.long)}
            def batch_decode(self, generated, **kwargs):
                return [str(i) for i in range(len(generated))]
        class Model:
            def generate(self, **kwargs):
                self.arguments = kwargs
                return torch.zeros(kwargs['input_features'].shape[0], 4, dtype=torch.long)
        with tempfile.TemporaryDirectory() as temporary:
            paths = []
            for i in range(3):
                path = Path(temporary)/f'{i}.wav'
                with wave.open(str(path), 'wb') as stream:
                    stream.setparams((1, 2, 16000, 160, 'NONE', 'not compressed'))
                    stream.writeframes(b'\0\0' * 160)
                paths.append(path)
            model = Model()
            backend = WhisperBackend(Processor(), model, 'cpu', batch_size=4)
            self.assertEqual(backend.transcribe_batch(paths), ['0', '1', '2'])
            self.assertTrue(calls[0]['return_attention_mask'])
            self.assertEqual(model.arguments['attention_mask'].shape[0], 3)
            self.assertEqual(model.arguments['language'], 'el')
            self.assertFalse(model.arguments['do_sample'])


if __name__ == '__main__':
    unittest.main()
