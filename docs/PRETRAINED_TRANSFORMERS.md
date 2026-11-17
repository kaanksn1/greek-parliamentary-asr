# Standard Whisper and Qwen3-ASR inference

Owner integration prepared on 18 November 2026. These adapters perform
**inference without additional group training**. Adnan owns the separate
Whisper fine-tuning implementation. This document does not claim a completed
benchmark or a measured runtime for Qwen.

## Shared interface

Each module exports
`load_backend(*, model_id, revision, device, model_dir=None)`. Load it once;
call the returned object's `transcribe(Path)` for each audio recording.
The result is a string, including an empty string when the model predicts
no text. The object's `settings` records its decoding/input policy.

`revision` must be a full 40-character checkpoint commit SHA. The optional
`model_dir` is a download/cache destination: the adapter still requests the
exact revision and does not trust arbitrary existing local weights. Both
adapters download JSON/text/safetensors resources, load the local pinned
snapshot, and request safetensors. Explicit `cuda`/`cuda:0` requests fail when
CUDA/ROCm is unavailable. `cpu` must be selected explicitly. No quantization,
FlashAttention, vLLM or context vocabulary is added to these standard runs.
ROCm uses PyTorch's `cuda` device interface.

## Whisper

Module: `greek_asr/pretrained/whisper.py`. Initial model:
`openai/whisper-small`, revision in `configs/model_sources.lock.json`.
Greedy transcription uses `language="el"`, `task="transcribe"`, one beam,
no sampling, float32 and eager attention. No reference text or prompt IDs
are supplied. Generation length uses the checkpoint generation configuration.

The adapter accepts HParl's original mono PCM16 16 kHz WAV and reads the
source bytes without rewriting/resampling the file. Empty, truncated or
wrong-format audio is an error. Model feature extraction is in memory.
Audio longer than 30 seconds is not silently truncated: all features and
the attention mask are passed to timestamp-based long-form generation.
Short audio uses the required 30-second feature padding. Timestamp tokens
are removed by the tokenizer's ordinary special-token decoding.

Implementation follows the official [Transformers Whisper documentation](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/whisper),
including its long-form generation example. The existing three-clip Whisper
startup smoke used its own shorter token cap; it is historical runtime
evidence, not a score from this new general adapter.

## Qwen3-ASR

Module: `greek_asr/pretrained/qwen.py`. Initial model:
`Qwen/Qwen3-ASR-0.6B`, revision in the same checkpoint lock. The original
checkpoint is used through `qwen-asr==0.0.6`, not the separately released
native-Transformers `-hf` conversion. The official package pins
`transformers==4.57.6` and `accelerate==1.12.0`; its remaining runtime
dependencies belong in the separately resolved environment lock.

The Transformers package backend uses float32/eager attention, batch limit
1, its documented 512-new-token default, one beam and no sampling. Calls
provide only the audio path, forced language `Greek`, empty context and
`return_time_stamps=False`. No forced-aligner model is loaded. The upstream
package handles its audio decoding/chunking in memory. Its finite generation
cap is part of the recorded baseline and must not be changed in response to
official-test scores. Missing/multiple/non-string results are errors rather
than silently omitted predictions.

Primary references: [official Qwen quickstart](https://github.com/QwenLM/Qwen3-ASR),
[package dependency declarations](https://github.com/QwenLM/Qwen3-ASR/blob/main/pyproject.toml),
[official wrapper implementation](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_asr.py)
and [model card](https://huggingface.co/Qwen/Qwen3-ASR-0.6B).

## Verification and remaining runtime work

`tests/test_pretrained_transformers.py` covers pinned one-time loading,
audio-only Greek decoding, empty/malformed results, explicit device failure,
unaltered source PCM, and the >30-second Whisper input path. Seven tests
passed without downloading models or importing their heavy dependencies.
These dependency doubles verify integration contracts; they are not evidence
that Qwen or the new adapter has run on the target GPU. Actual central
train-only runtime logs must be recorded separately before full experiments.
No development/test audio was used to implement or test these adapters.
