"""Semantic regression tests for real CSS10's IEEE-float input format."""
from __future__ import annotations

import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from greek_asr.audio import read_wav_bytes
from greek_asr.infer import load_manifest


def wav_bytes(samples, *, code=3, rate=22050, extra=b""):
    if code == 3:
        payload, width = struct.pack("<" + "f" * len(samples), *samples), 4
    elif code == 1:
        payload, width = struct.pack("<" + "i" * len(samples), *samples), 4
    else:
        payload, width = bytes(samples), 1
    fmt = struct.pack("<HHIIHH", code, 1, rate, rate * width, width, width * 8)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + extra + b"data" + struct.pack("<I", len(payload)) + payload
    return b"RIFF" + struct.pack("<I", len(body)) + body


class AudioInputContractTests(unittest.TestCase):
    def test_float32_and_pcm32_encode_equal_amplitudes_with_different_bytes(self):
        floating = read_wav_bytes(wav_bytes([0., .5, -.5, .25]))
        integer = read_wav_bytes(wav_bytes([0, 2**30, -2**30, 2**29], code=1))
        self.assertNotEqual(floating.payload, integer.payload)
        self.assertEqual(floating.encoding, "ieee_float")
        self.assertEqual(integer.encoding, "integer_pcm")
        np.testing.assert_array_equal(floating.mono_float32(), integer.mono_float32())
        self.assertEqual(floating.num_frames, 4)
        self.assertEqual(floating.sampling_rate, 22050)

    def test_float_samples_are_not_peak_normalized_or_clipped(self):
        actual = read_wav_bytes(wav_bytes([.01, -.02, 1.25])).mono_float32()
        np.testing.assert_allclose(actual, [.01, -.02, 1.25])

    def test_nonfinite_float_audio_is_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "nonfinite"):
                read_wav_bytes(wav_bytes([0., value])).mono_float32()

    def test_compressed_audio_is_not_misread_as_integer_pcm(self):
        with self.assertRaisesRegex(ValueError, "unsupported"):
            read_wav_bytes(wav_bytes([0, 1, 2, 3], code=6))

    def test_truncated_audio_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "truncated"):
            read_wav_bytes(wav_bytes([0., .5])[:-1])

    def test_odd_auxiliary_chunk_does_not_shift_the_audio(self):
        junk = b"JUNK" + struct.pack("<I", 3) + b"abc\0"
        actual = read_wav_bytes(wav_bytes([.25, -.25], extra=junk)).mono_float32()
        np.testing.assert_array_equal(actual, [.25, -.25])

    def test_runner_accepts_real_float_contract_without_exposing_gold_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = wav_bytes([0., .5, -.5])
            audio = root / "float.wav"
            audio.write_bytes(data)
            row = {"id": "css10_el:fixture", "source": "css10_el", "original_id": "fixture",
                   "source_split": "unsplit", "split": "train", "audio_path": "float.wav",
                   "audio_sha256": hashlib.sha256(data).hexdigest(), "sampling_rate": 22050,
                   "channels": 1, "num_frames": 3, "duration_seconds": 3/22050,
                   "reference": "must never be passed to a model"}
            manifest = root / "train.jsonl"
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
            result = load_manifest(manifest, root, "train")
            self.assertEqual(len(result), 1)
            self.assertNotIn("reference", result[0])
            self.assertEqual(audio.read_bytes(), data)
            with self.assertRaisesRegex(ValueError, "split mismatch|training-only"):
                load_manifest(manifest, root, "test")


if __name__ == "__main__":
    unittest.main()
