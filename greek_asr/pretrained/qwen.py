"""Pinned Qwen3-ASR package inference; no reference/context prompting or training."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any


class QwenBackend:
    def __init__(self, model: Any, batch_size: int = 1):
        self.model = model
        observed_attention = {}
        for name, module in model.model.named_modules():
            implementation = getattr(getattr(module, "config", None), "_attn_implementation", None)
            if isinstance(implementation, str):
                observed_attention[name or "<root>"] = implementation
        self.settings = {
            "language": "Greek", "context": "", "return_time_stamps": False,
            "dtype": "float32", "requested_attention": "eager",
            "observed_module_attention": observed_attention,
            "attention_note": "upstream composite submodules may retain their default SDPA implementation",
            "num_beams": 1,
            "do_sample": False, "max_inference_batch_size": batch_size,
            "max_new_tokens": 512, "backend": "qwen-asr transformers",
            "reference_or_context_prompt": False,
        }

    def transcribe(self, audio_path: Path) -> str:
        import torch

        audio_path = Path(audio_path)
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        with torch.inference_mode():
            results = self.model.transcribe(
                audio=str(audio_path), language="Greek", context="",
                return_time_stamps=False,
            )
        if len(results) != 1 or not isinstance(getattr(results[0], "text", None), str):
            raise ValueError("Qwen returned an invalid single-record transcription")
        return results[0].text


    def transcribe_batch(self, audio_paths: list[Path]) -> list[str]:
        import torch

        if not audio_paths:
            raise ValueError("Qwen batch must not be empty")
        paths = [Path(path) for path in audio_paths]
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
        with torch.inference_mode():
            results = self.model.transcribe(audio=[str(path) for path in paths], language="Greek", context="",
                                            return_time_stamps=False)
        if len(results) != len(paths) or any(not isinstance(getattr(result, "text", None), str) for result in results):
            raise ValueError("Qwen returned an invalid batch transcription count/type")
        return [result.text for result in results]


def load_backend(*, model_id: str, revision: str, device: str,
                 model_dir: Path | None = None, batch_size: int = 1) -> QwenBackend:
    if type(batch_size) is not int or not 1 <= batch_size <= 8:
        raise ValueError("Qwen batch_size must be an integer from 1 to 8")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be a full, immutable Hugging Face commit SHA")
    if not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", device):
        raise ValueError("device must be cpu, cuda or cuda:<index> (ROCm uses cuda)")
    import torch
    from huggingface_hub import snapshot_download
    from qwen_asr import Qwen3ASRModel

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA/ROCm device is unavailable; no CPU fallback")
    local = snapshot_download(
        repo_id=model_id, revision=revision, token=False,
        local_dir=str(model_dir) if model_dir is not None else None,
        allow_patterns=["*.json", "*.txt", "*.safetensors"],
    )
    model = Qwen3ASRModel.from_pretrained(
        local, dtype=torch.float32, device_map=device, attn_implementation="eager",
        local_files_only=True, use_safetensors=True,
        max_inference_batch_size=batch_size, max_new_tokens=512,
    )
    model.model.eval()
    model.model.generation_config.do_sample = False
    model.model.generation_config.num_beams = 1
    return QwenBackend(model, batch_size)
