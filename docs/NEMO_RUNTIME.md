# Canary and Parakeet runtime

The owner integration in `greek_asr/pretrained/nemo.py` covers the standard
pretrained Canary and Parakeet baselines. It does not train or change their
decoders. The common runner provides the exact revision from
`configs/model_sources.lock.json`.

| Model | Producer checkpoint file | Inference policy |
|---|---|---|
| `nvidia/canary-1b-v2` | `canary-1b-v2.nemo` | Greek source and Greek target (`el`, `el`); checkpoint decoding defaults |
| `nvidia/parakeet-tdt-0.6b-v3` | `parakeet-tdt-0.6b-v3.nemo` | Producer's automatic language detection; checkpoint decoding defaults |

The files were verified in the Hugging Face file listings at the pinned
commits on 18 November 2026. `hf_hub_download` selects that commit and file;
`ASRModel.restore_from` restores the resulting local archive. This avoids
silently loading the current `main` through an unversioned `from_pretrained`.
One model is restored per run, inference mode is used, and one recording is
transcribed at a time. The adapter receives only an audio path. Source audio
is not overwritten. Both models expect mono 16 kHz input; the common manifest
validation must check that contract before inference.

Canary is also a translation model, so both language parameters are explicit.
Parakeet v3 detects language itself; Canary-style prompt arguments are not
passed to it. Returned Hypothesis objects and plain text are supported; a
missing, multiple or non-text result fails visibly. An actual empty text
hypothesis is retained for evaluation.

## Environment and status

As of 18 November 2026, the implementation has dependency-free boundary tests.
Those tests verify checkpoint selection, task/language parameters, inference
cohort shape and failure propagation. They are not evidence of model accuracy
or a successful GPU run. The separate runtime must pass an import check and a
train-only audio smoke before either baseline is reported as runtime-verified.

The current exact PyPI release is **`nemo_toolkit==3.0.0`**. Its official
metadata declares Python >=3.10 and torch >=2.6.0; the `[asr]` extra adds the
audio, tokenizer, dataset and Lightning dependencies. Install NeMo in its own
environment with the project's exact runtime pins, preserving the selected
PyTorch backend. NVIDIA CUDA extras or a CUDA torch build are not a replacement
for the existing AMD ROCm build. The base package also declares `cuda-bindings`;
package availability alone does not establish AMD execution support.

The older `nemo_toolkit==2.5.3` ASR extra requires NumPy <2 and Transformers
~=4.53.0. Installing it into the previous Whisper environment (NumPy 2.5.3,
Transformers 4.57.6) would change that environment. It is therefore not an
in-place setup recipe. Full AMD compatibility and any dependency conflict must
be recorded from the actual isolated installation, not inferred from these
metadata checks. Do not downgrade the shared BANKING77 environment.

## Primary sources

- [Canary producer card and Greek transcription API](https://huggingface.co/nvidia/canary-1b-v2)
- [Parakeet producer card and language detection API](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- [NeMo 3.0.0 package metadata](https://pypi.org/pypi/nemo_toolkit/3.0.0/json)
- [NeMo 2.5.3 package metadata](https://pypi.org/pypi/nemo_toolkit/2.5.3/json)
- [NeMo restore API implementation](https://github.com/NVIDIA-NeMo/NeMo/blob/v2.5.3/nemo/core/classes/modelPT.py)
- [Canary pinned source tree](https://huggingface.co/nvidia/canary-1b-v2/tree/d455706339a6b32e1aa40f82c713a482a0c938e2)
- [Parakeet pinned source tree](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/tree/541d1f99c6b0c3cd0b11a95167540bb8edefd82b)
