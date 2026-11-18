"""Standard, revision-pinned NeMo inference for Canary and Parakeet.

Imports are lazy so the common data/evaluation tools do not require NeMo.
No transcript, label or scoring input is accepted by either backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any


MODEL_FILES = {
    "nvidia/canary-1b-v2": "canary-1b-v2.nemo",
    "nvidia/parakeet-tdt-0.6b-v3": "parakeet-tdt-0.6b-v3.nemo",
}


def _import_runtime() -> tuple[Any, Any, Any]:
    try:
        import torch
        from huggingface_hub import hf_hub_download
        from nemo.collections.asr.models import ASRModel
    except ImportError as exc:
        raise RuntimeError(
            "NeMo ASR runtime could not be imported. Use the project's isolated "
            "NeMo environment and its pinned dependencies; see docs/NEMO_RUNTIME.md. "
            f"Original import error: {exc}. "
            "Do not substitute an empty prediction for this failure."
        ) from exc
    return torch, hf_hub_download, ASRModel


def _disable_parakeet_cuda_graphs(config: Any) -> None:
    """Disable an NVIDIA-only execution optimization, keeping greedy TDT."""
    from omegaconf import OmegaConf

    OmegaConf.update(
        config, "decoding.greedy.use_cuda_graph_decoder", False, force_add=True
    )


@dataclass
class NemoBackend:
    """One restored model reused for all utterances in a run."""

    model_id: str
    revision: str
    device: str
    checkpoint_path: str
    _model: Any
    _torch: Any
    _cuda_graphs_disabled: bool = False

    @property
    def settings(self) -> dict[str, Any]:
        settings = {
            "runtime": "nemo",
            "checkpoint_file": MODEL_FILES[self.model_id],
            "batch_size": 1,
            "source_lang": "el" if self.model_id == "nvidia/canary-1b-v2" else "auto",
            "target_lang": "el" if self.model_id == "nvidia/canary-1b-v2" else "auto",
            "decoder": "checkpoint defaults",
        }
        if self._cuda_graphs_disabled:
            settings["use_cuda_graph_decoder"] = False
            settings["execution_override"] = "NVIDIA CUDA Graph optimization disabled on ROCm; same greedy TDT"
        return settings

    def transcribe(self, audio_path: Path) -> str:
        path = Path(audio_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)

        # Canary is multitask: both language tokens are required to select
        # Greek transcription rather than Greek-to-English translation.
        options: dict[str, Any] = {"batch_size": 1}
        if self.model_id == "nvidia/canary-1b-v2":
            options.update(source_lang="el", target_lang="el")
        # Parakeet v3 detects language itself; it has no Canary language prompt.
        with self._torch.inference_mode():
            results = self._model.transcribe([str(path)], **options)
        if not isinstance(results, (list, tuple)) or len(results) != 1:
            raise RuntimeError(
                f"NeMo returned an unexpected number/shape of predictions for {path.name}"
            )
        result = results[0]
        text = result if isinstance(result, str) else getattr(result, "text", None)
        if not isinstance(text, str):
            raise RuntimeError(f"NeMo returned a non-text prediction for {path.name}")
        # An actual empty hypothesis is retained for corpus scoring. A failed
        # or missing result above is never silently converted into an empty one.
        return text


def load_backend(
    *, model_id: str, revision: str, device: str, model_dir: Path | None = None
) -> NemoBackend:
    """Download the exact producer archive and restore it once.

    ``model_dir`` is an optional Hugging Face cache directory, not an arbitrary
    local checkpoint override. The original checkpoint's decoding is retained.
    """
    if model_id not in MODEL_FILES:
        raise ValueError(f"Unsupported NeMo model: {model_id}")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be a full 40-character Hugging Face commit SHA")
    torch, download, model_class = _import_runtime()
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested {device}, but PyTorch cannot access a GPU; refusing CPU fallback"
        )
    checkpoint_path = download(
        repo_id=model_id,
        filename=MODEL_FILES[model_id],
        revision=revision,
        cache_dir=str(model_dir) if model_dir is not None else None,
    )
    restore_options: dict[str, Any] = {}
    rocm_parakeet = model_id == "nvidia/parakeet-tdt-0.6b-v3" and bool(
        getattr(getattr(torch, "version", None), "hip", None)
    )
    if rocm_parakeet:
        # NeMo 3's default TDT decoder probes NVIDIA's driver during model
        # construction. ROCm must disable that execution optimization before
        # restore, not after an already-failed model constructor. NeMo extracts
        # only YAML for return_config, leaving source weights untouched.
        config = model_class.restore_from(
            restore_path=checkpoint_path, map_location="cpu", return_config=True
        )
        _disable_parakeet_cuda_graphs(config)
        restore_options["override_config_path"] = config
    model = model_class.restore_from(
        restore_path=checkpoint_path, map_location=torch_device, **restore_options
    )
    model.eval()
    return NemoBackend(
        model_id=model_id,
        revision=revision,
        device=device,
        checkpoint_path=str(checkpoint_path),
        _model=model,
        _torch=torch,
        _cuda_graphs_disabled=rocm_parakeet,
    )
