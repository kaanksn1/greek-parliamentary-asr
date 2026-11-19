"""Shared original-WAV input contract for the pinned Python 3.12 runtimes.

RIFF format tag, not sample width alone, distinguishes integer PCM32 from IEEE
float32. Source bytes are never converted on disk. Supported corpus encodings:
PCM16, PCM32, and IEEE float32, little-endian, uncompressed RIFF/WAVE.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WavAudio:
    format_tag: int
    sampling_rate: int
    channels: int
    sample_width_bytes: int
    num_frames: int
    payload: bytes

    @property
    def encoding(self) -> str:
        return "ieee_float" if self.format_tag == 3 else "integer_pcm"

    def mono_float32(self):
        """Decode true amplitudes, without peak scaling, clipping or resampling."""
        import numpy as np

        if self.channels != 1:
            raise ValueError(f"expected mono audio, found {self.channels} channels")
        if self.format_tag == 3:
            values = np.frombuffer(self.payload, dtype="<f4").astype(np.float32, copy=True)
        elif self.sample_width_bytes == 2:
            values = np.frombuffer(self.payload, dtype="<i2").astype(np.float32) / 32768.
        else:
            values = np.frombuffer(self.payload, dtype="<i4").astype(np.float32) / 2147483648.
        if not np.isfinite(values).all():
            raise ValueError("audio contains nonfinite samples")
        return values


def read_wav_bytes(data: bytes, label: str = "audio") -> WavAudio:
    if data.startswith(b"version https://git-lfs.github.com/spec/"):
        raise ValueError(f"{label}: Git LFS pointer instead of audio (run git lfs pull)")
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"{label}: unsupported or malformed WAV file")
    if struct.unpack_from("<I", data, 4)[0] + 8 != len(data):
        raise ValueError(f"{label}: truncated WAV payload or inconsistent RIFF length")
    position, fmt, payload = 12, None, None
    while position < len(data):
        if position + 8 > len(data):
            raise ValueError(f"{label}: truncated WAV chunk header")
        tag, size = struct.unpack_from("<4sI", data, position)
        start, end = position + 8, position + 8 + size
        if end > len(data):
            raise ValueError(f"{label}: truncated WAV payload")
        if tag == b"fmt ":
            if fmt is not None or size < 16:
                raise ValueError(f"{label}: invalid or duplicate WAV format chunk")
            fmt = struct.unpack_from("<HHIIHH", data, start)
        elif tag == b"data":
            if payload is not None:
                raise ValueError(f"{label}: duplicate WAV data chunk")
            payload = data[start:end]
        position = end + size % 2
        if position > len(data):
            raise ValueError(f"{label}: missing odd-chunk padding byte")
    if fmt is None or payload is None:
        raise ValueError(f"{label}: missing WAV format/data chunk")
    code, channels, rate, byte_rate, block_align, bits = fmt
    if (code, bits) not in ((1, 16), (1, 32), (3, 32)):
        raise ValueError(f"{label}: unsupported PCM sample width or encoding ({bits}-bit, format tag {code}); "
                         "supported: integer PCM16/PCM32 and IEEE float32")
    if channels < 1 or rate < 1 or block_align != channels * (bits // 8) or byte_rate != rate * block_align:
        raise ValueError(f"{label}: invalid WAV format properties")
    if not payload or len(payload) % block_align:
        raise ValueError(f"{label}: empty or truncated WAV payload")
    return WavAudio(code, rate, channels, bits // 8, len(payload) // block_align, payload)


def read_wav(path: Path) -> WavAudio:
    return read_wav_bytes(Path(path).read_bytes(), str(path))
