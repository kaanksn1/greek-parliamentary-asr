"""Format-aware WAV probing and signal statistics of the data pipeline (audit schema v2).

Regression for the CSS10 bug: the real clips are IEEE float32 (RIFF format
tag 3). Python 3.12's ``wave`` rejects them, and the former signal_stats read
every 4-byte sample as integer PCM32, which turned about -27.9 dBFS speech
into about -6 dBFS. The format tag, not the sample width, decides decoding.
"""
from __future__ import annotations

import hashlib
import io
import math
import struct
import sys
import unittest
import wave

from greek_asr.data import audit
from greek_asr.data import common
from greek_asr.data.common import DataError, probe_wav_bytes
from tests.data_fixtures import float32_wav_bytes, pcm, pcm32_wav_bytes, wav_bytes

SIGNAL = {"frame_seconds": 0.025, "silence_frame_dbfs": -45.0, "clip_threshold": 0.999}


def stats(data: bytes) -> dict:
    measured = probe_wav_bytes(data)
    return audit.signal_stats(measured["_payload"], measured["sample_width_bytes"], measured["channels"],
                              measured["sampling_rate"], SIGNAL, format_tag=measured["format_tag"])


def sine(amplitude: float, seconds: float = 1.0, rate: int = 22050, frequency: float = 22050 / 40) -> list[float]:
    """40 samples per period, so the exact positive and negative peaks are sampled."""
    return [amplitude * math.sin(2 * math.pi * frequency * n / rate) for n in range(int(seconds * rate))]


class FormatTagTests(unittest.TestCase):
    def test_float32_and_pcm32_with_the_same_width_are_decoded_by_format_tag(self):
        floating = probe_wav_bytes(float32_wav_bytes([0.5] * 100, 16000))
        integer = probe_wav_bytes(pcm32_wav_bytes([1 << 30] * 100, 16000))  # 2**30 / 2**31 = 0.5
        self.assertEqual((floating["sample_width_bytes"], integer["sample_width_bytes"]), (4, 4))
        self.assertEqual((floating["format_tag"], floating["encoding"]), (3, "ieee_float32le"))
        self.assertEqual((integer["format_tag"], integer["encoding"]), (1, "pcm_s32le"))
        for measured in (floating, integer):
            samples = audit.decode_samples(measured["_payload"], measured["format_tag"], 4)
            self.assertEqual(set(samples.tolist()), {0.5})
        # the former bug: float bytes reinterpreted as int32 give a different "amplitude"
        misread = audit.decode_samples(floating["_payload"], 1, 4)
        self.assertAlmostEqual(float(misread[0]), 0x3F000000 / 2 ** 31)
        self.assertNotAlmostEqual(float(misread[0]), 0.5, places=3)
        self.assertAlmostEqual(stats(float32_wav_bytes([0.5] * 2205))["rms_dbfs"], 20 * math.log10(0.5), places=3)
        self.assertAlmostEqual(stats(pcm32_wav_bytes([1 << 30] * 1600))["rms_dbfs"], 20 * math.log10(0.5), places=3)

    def test_pcm16_and_pcm32_remain_supported_with_integer_scaling(self):
        pcm16 = probe_wav_bytes(wav_bytes(struct.pack("<4h", 16384, -16384, 16384, -16384)))
        self.assertEqual((pcm16["format_tag"], pcm16["encoding"], pcm16["num_frames"]), (1, "pcm_s16le", 4))
        self.assertEqual(audit.decode_samples(pcm16["_payload"], 1, 2).tolist(), [0.5, -0.5, 0.5, -0.5])
        pcm32 = probe_wav_bytes(pcm32_wav_bytes([-(1 << 31), (1 << 31) - 1, 0]))
        decoded = audit.decode_samples(pcm32["_payload"], 1, 4).tolist()
        self.assertEqual(decoded[0], -1.0)
        self.assertAlmostEqual(decoded[1], 1.0, places=8)
        self.assertEqual(pcm32["duration_seconds"], 3 / 16000)

    def test_known_float_amplitudes_give_exact_rms_peak_clipping_and_silence(self):
        quiet = stats(float32_wav_bytes(sine(0.1)))  # rms = 0.1/sqrt(2), peak 0.1
        self.assertAlmostEqual(quiet["rms_dbfs"], 20 * math.log10(0.1 / math.sqrt(2)), places=2)
        self.assertAlmostEqual(quiet["peak_dbfs"], -20.0, places=2)
        self.assertEqual(quiet["clip_fraction"], 0.0)
        self.assertEqual(quiet["speech_fraction"], 1.0)
        # no gain/peak normalization: a -40 dBFS signal stays at -40 dBFS
        self.assertAlmostEqual(stats(float32_wav_bytes(sine(0.01)))["peak_dbfs"], -40.0, places=2)
        # float values may exceed full scale; they count as clipped and are not rescaled
        hot = stats(float32_wav_bytes([1.5, -1.5] * 1000))
        self.assertEqual(hot["clip_fraction"], 1.0)
        self.assertAlmostEqual(hot["peak_dbfs"], 20 * math.log10(1.5), places=3)
        silent = stats(float32_wav_bytes([0.0] * 22050))
        self.assertLessEqual(silent["peak_dbfs"], -199.0)
        self.assertEqual(silent["speech_fraction"], 0.0)
        edges = stats(float32_wav_bytes([0.0] * 22050 + sine(0.2) + [0.0] * 11025))
        self.assertAlmostEqual(edges["leading_silence_seconds"], 1.0, delta=0.03)
        self.assertAlmostEqual(edges["trailing_silence_seconds"], 0.5, delta=0.03)

    def test_known_pcm16_amplitude_statistics(self):
        values = [int(round(0.25 * 32767 * math.sin(2 * math.pi * 400 * n / 16000))) for n in range(16000)]
        result = stats(wav_bytes(struct.pack(f"<{len(values)}h", *values)))
        self.assertAlmostEqual(result["rms_dbfs"], 20 * math.log10(0.25 / math.sqrt(2)), places=2)
        self.assertEqual(result["clip_fraction"], 0.0)
        clipped = stats(wav_bytes(pcm(1.0, kind="clipped")))
        self.assertGreater(clipped["clip_fraction"], 0.9)

    def test_css10_float32_is_read_without_the_wave_module_on_python_312(self):
        data = float32_wav_bytes(sine(0.2, 0.5), 22050)
        measured = probe_wav_bytes(data, "css10")
        self.assertEqual((measured["sampling_rate"], measured["channels"], measured["num_frames"]), (22050, 1, 11025))
        self.assertEqual(measured["audio_sha256"], hashlib.sha256(data).hexdigest())
        self.assertNotIn("wave", vars(common), "the data probe must not depend on the wave module")
        if sys.version_info < (3, 15):  # documents the original Python 3.12 failure
            with self.assertRaises(wave.Error):
                wave.open(io.BytesIO(data), "rb")

    def test_pcm_hash_is_the_exact_data_chunk_like_the_former_wave_probe(self):
        data = wav_bytes(pcm(0.3, seed=4))
        with wave.open(io.BytesIO(data), "rb") as stream:
            frames = stream.readframes(stream.getnframes())
        self.assertEqual(probe_wav_bytes(data)["pcm_sha256"], hashlib.sha256(frames).hexdigest())


class VisibleFailureTests(unittest.TestCase):
    def test_nan_and_inf_float_samples_fail(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            data = float32_wav_bytes([0.1, bad, 0.1])
            with self.assertRaisesRegex(DataError, "NaN/Inf"):
                probe_wav_bytes(data)
            payload = struct.pack("<3f", 0.1, bad, 0.1)
            with self.assertRaisesRegex(DataError, "NaN/Inf"):
                audit.decode_samples(payload, 3, 4)

    def test_truncated_or_inconsistent_wav_fails(self):
        data = float32_wav_bytes(sine(0.1, 0.1))
        for broken in (data[:-3], data[:-4], data[:200], data[:30]):
            with self.assertRaises(DataError):
                probe_wav_bytes(broken)
        lying = data[:4] + struct.pack("<I", len(data) + 100) + data[8:]
        with self.assertRaisesRegex(DataError, "truncated|inconsistent"):
            probe_wav_bytes(lying)

    def test_lfs_pointer_and_unsupported_encodings_fail(self):
        with self.assertRaisesRegex(DataError, "LFS pointer"):
            probe_wav_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"0" * 64 + b"\nsize 9\n")
        pcm8 = bytearray(wav_bytes(b"\x00\x00\x00\x00"))
        struct.pack_into("<HHIIHH", pcm8, 20, 1, 1, 16000, 32000, 2, 8)  # claims 8-bit PCM
        with self.assertRaises(DataError):
            probe_wav_bytes(bytes(pcm8))
        with self.assertRaisesRegex(DataError, "unsupported"):
            audit.decode_samples(b"\x00" * 6, 1, 3)
        with self.assertRaisesRegex(DataError, "unsupported"):
            audit.decode_samples(b"\x00" * 8, 3, 8)


if __name__ == "__main__":
    unittest.main()
