"""Standard inference for a published, Greek fine-tuned wav2vec 2.0 + CTC checkpoint.

Baseline: ``lighteternal/wav2vec2-large-xlsr-53-greek`` at the immutable revision
in ``configs/model_sources.lock.json``. Its model card (Apache-2.0) states a
Greek (``el``) XLSR-53 encoder with a character CTC head fine-tuned on Common
Voice Greek and CSS10 Greek, 16 kHz input. It is run without any training by
our group. Note that its producer used CSS10 Greek, our training-only secondary
source; the HParl test set is not among its declared training data.

Path: original 16 kHz audio -> checkpoint feature extractor (per-utterance
normalization) -> Wav2Vec2ForCTC -> greedy CTC over all frames of the single
utterance (batch size 1, so no padded frame exists) -> tokenizer characters
(``|`` word delimiter -> space) -> common prediction JSONL via the shared
runner ``greek_asr.infer`` (backend name ``wav2vec``). No language model,
no reference text or prompt is ever an input.

``HFCTCBackend`` is reused for the published HuBERT CTC candidate
(``greek_asr.models.hubert``).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from greek_asr.models import common, ctc_utils

MODEL_FILES = ["config.json", "preprocessor_config.json", "vocab.json", "tokenizer_config.json",
               "special_tokens_map.json", "added_tokens.json", "model.safetensors", "pytorch_model.bin"]


def validate_load_options(revision: str, device: str) -> None:
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be a full, immutable Hugging Face commit SHA")
    if not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", device):
        raise ValueError("device must be cpu, cuda or cuda:<index> (ROCm uses cuda)")


class HFCTCBackend:
    """Single-utterance greedy CTC transcription with a Hugging Face ``*ForCTC`` model."""

    def __init__(self, feature_extractor: Any, tokenizer: Any, model: Any, device: str, *, model_label: str):
        self.feature_extractor, self.tokenizer, self.model, self.device = feature_extractor, tokenizer, model, device
        self.sampling_rate = int(feature_extractor.sampling_rate)
        blank = model.config.pad_token_id
        if blank is None or tokenizer.convert_ids_to_tokens(blank) != tokenizer.pad_token:
            raise ValueError("model pad_token_id must be the tokenizer's pad (CTC blank) token")
        # Published CTC tokenizers often register <s>/</s> as added tokens beyond the
        # output layer (e.g. the Greek wav2vec: 55 logits, 57 tokenizer entries); only
        # the base vocabulary must be covered by the model's logits.
        if model.config.vocab_size < tokenizer.vocab_size:
            raise ValueError(f"model vocab_size {model.config.vocab_size} < tokenizer base vocabulary "
                             f"{tokenizer.vocab_size}")
        self.blank_id = int(blank)
        self.settings = {
            "model_class": type(model).__name__, "model": model_label, "dtype": "float32",
            "batch_size": 1, "sampling_rate": self.sampling_rate,
            "feature_extractor": {"do_normalize": bool(getattr(feature_extractor, "do_normalize", False)),
                                  "return_attention_mask": bool(feature_extractor.return_attention_mask)},
            "decoding": "greedy CTC: per-frame argmax, merge repeats, drop blank; no language model",
            "word_delimiter": getattr(tokenizer, "word_delimiter_token", None),
            "audio_input": "original audio, resampled in memory only if its rate differs",
            "long_audio": "whole utterance in one forward pass, no truncation",
            "reference_or_context_prompt": False,
        }

    def transcribe(self, audio_path: Path) -> str:
        import torch

        waveform = common.load_waveform(Path(audio_path), target_sample_rate=self.sampling_rate)
        if waveform.numel() == 0:
            raise ValueError("audio has no samples")
        inputs = self.feature_extractor(waveform.numpy(), sampling_rate=self.sampling_rate, return_tensors="pt",
                                        return_attention_mask=bool(self.feature_extractor.return_attention_mask))
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            logits = self.model(**inputs).logits
        if logits.shape[0] != 1:
            raise ValueError("expected a single-utterance batch")
        ids = ctc_utils.ctc_greedy_decode(logits, [logits.shape[1]], self.blank_id)[0]
        # Repeats are already merged by the CTC collapse; grouping again would merge
        # genuine double letters that a blank separated (e.g. "λλ").
        text = self.tokenizer.decode(ids, group_tokens=False, skip_special_tokens=True)
        return " ".join(text.split())


def load_ctc_backend(*, model_id: str, revision: str, device: str, model_dir: Path | None, model_class: str):
    validate_load_options(revision, device)
    import torch
    from huggingface_hub import snapshot_download
    from transformers import HubertForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA/ROCm device is unavailable; no CPU fallback")
    classes = {"Wav2Vec2ForCTC": Wav2Vec2ForCTC, "HubertForCTC": HubertForCTC}
    local = snapshot_download(repo_id=model_id, revision=revision, token=False,
                              local_dir=str(model_dir) if model_dir is not None else None,
                              allow_patterns=MODEL_FILES)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(local, local_files_only=True)
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(local, local_files_only=True)
    model = classes[model_class].from_pretrained(local, local_files_only=True, torch_dtype=torch.float32)
    if type(model).__name__ != model_class:
        raise ValueError(f"{model_id} did not load as {model_class}")
    return HFCTCBackend(feature_extractor, tokenizer, model.to(device).eval(), device,
                        model_label=f"{model_id}@{revision}")


def load_backend(*, model_id: str, revision: str, device: str, model_dir: Path | None = None) -> HFCTCBackend:
    """Loader used by ``python -m greek_asr.infer --backend wav2vec``."""
    return load_ctc_backend(model_id=model_id, revision=revision, device=device, model_dir=model_dir,
                            model_class="Wav2Vec2ForCTC")
