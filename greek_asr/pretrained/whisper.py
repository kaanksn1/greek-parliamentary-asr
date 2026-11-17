"""Pinned multilingual Whisper inference; audio-only input, without fine-tuning."""
from __future__ import annotations

import re
import wave
from pathlib import Path
from typing import Any


def _validate_load_options(revision: str, device: str) -> None:
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be a full, immutable Hugging Face commit SHA")
    if not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", device):
        raise ValueError("device must be cpu, cuda or cuda:<index> (ROCm uses cuda)")


def _read_pcm16(audio_path: Path) -> tuple[bytes, int]:
    """Read original HParl PCM bytes; never overwrite/resample source audio."""
    with wave.open(str(audio_path), "rb") as handle:
        if (handle.getframerate(), handle.getnchannels(), handle.getsampwidth(),
                handle.getcomptype()) != (16000, 1, 2, "NONE"):
            raise ValueError("Whisper adapter requires original mono PCM16 16 kHz WAV")
        frames = handle.getnframes()
        if frames == 0:
            raise ValueError("audio has no frames")
        pcm = handle.readframes(frames)
    if len(pcm) != frames * 2:
        raise ValueError("audio PCM payload is truncated")
    return pcm, frames


class WhisperBackend:
    def __init__(self, processor: Any, model: Any, device: str):
        self.processor, self.model, self.device = processor, model, device
        self.settings = {
            "language": "el", "task": "transcribe", "dtype": "float32",
            "attention": "eager", "num_beams": 1, "do_sample": False,
            "batch_size": 1, "audio_input": "original mono PCM16 16000 Hz WAV",
            "long_audio": "untruncated features and timestamp-based generation above 30 s",
            "generation_length": "checkpoint generation_config default",
            "reference_or_context_prompt": False,
        }

    def transcribe(self, audio_path: Path) -> str:
        import numpy as np
        import torch

        pcm, frames = _read_pcm16(Path(audio_path))
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        long_audio = frames > 30 * 16000
        inputs = self.processor(
            audio, sampling_rate=16000, return_tensors="pt",
            return_attention_mask=True, truncation=False,
            padding="longest" if long_audio else "max_length",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs, language="el", task="transcribe", num_beams=1,
                do_sample=False, return_timestamps=long_audio,
            )
        texts = self.processor.batch_decode(generated, skip_special_tokens=True)
        if len(texts) != 1 or not isinstance(texts[0], str):
            raise ValueError("Whisper returned an invalid single-record transcription")
        return texts[0]


def load_backend(*, model_id: str, revision: str, device: str,
                 model_dir: Path | None = None) -> WhisperBackend:
    """Load exactly one pinned checkpoint. Explicit CUDA failure never falls back."""
    _validate_load_options(revision, device)
    import torch
    from huggingface_hub import snapshot_download
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA/ROCm device is unavailable; no CPU fallback")
    local = snapshot_download(
        repo_id=model_id, revision=revision, token=False,
        local_dir=str(model_dir) if model_dir is not None else None,
        allow_patterns=["*.json", "*.txt", "*.safetensors"],
    )
    processor = WhisperProcessor.from_pretrained(local, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        local, local_files_only=True, torch_dtype=torch.float32,
        attn_implementation="eager", use_safetensors=True,
    ).to(device).eval()
    return WhisperBackend(processor, model, device)
