# Canary and Parakeet runtime

The owner integration in `greek_asr/pretrained/nemo.py` covers the standard
pretrained Canary and Parakeet baselines. It does not train or change their
decoding algorithms. Parakeet's NVIDIA-only CUDA Graph execution optimization
is disabled on ROCm, as recorded below. The common runner provides the exact revision from
`configs/model_sources.lock.json`.

| Model | Producer checkpoint file | Inference policy |
|---|---|---|
| `nvidia/canary-1b-v2` | `canary-1b-v2.nemo` | Greek source and Greek target (`el`, `el`); checkpoint decoding defaults |
| `nvidia/parakeet-tdt-0.6b-v3` | `parakeet-tdt-0.6b-v3.nemo` | Producer's automatic language detection and greedy TDT; NVIDIA CUDA Graph optimization disabled on ROCm |

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

As of 18 November 2026, the implementation has seven dependency-free boundary tests.
Those tests verify checkpoint selection, task/language parameters, inference
cohort shape and failure propagation. They are not evidence of model accuracy
or a successful GPU inference run. The separate native WSL runtime passed an
actual `ASRModel` import and `pip check`; PyTorch sees the AMD Radeon RX 7800 XT
with `HSA_ENABLE_DXG_DETECTION=1`. Canary also completed inference on all three
original training-pilot recordings; its real record and predictions are in
`results/runtime_smoke/owner_canary_train/`. Parakeet's corrected run also
completed all three in `results/runtime_smoke/owner_parakeet_train_verified/`.
Both records contain the pinned checkpoint, environment, seed, complete ID
cohort, source/code/prediction hashes, runtime and memory observations. These
are runtime checks, not WER/CER benchmarks or training results. No dev/test
audio or references were used. Three short clips do not establish a full-data
training budget, long-form decoding performance or statistical sufficiency.

The first Parakeet restore encountered a real ROCm incompatibility: NeMo 3's
default greedy TDT setup probes `cuda.bindings` for NVIDIA CUDA Graph support
and cannot find an NVIDIA driver on this AMD system. The outer restore error
was an abstract `ASRModel` TypeError, but the nested traceback identified the
CUDA driver probe. The fix reads the producer YAML through the official
`restore_from(return_config=True)` API, sets
`decoding.greedy.use_cuda_graph_decoder=false` with OmegaConf, and restores
the same pinned checkpoint with this config override. The flag is a documented
NeMo execution option; automatic language detection, weights, greedy TDT
algorithm and audio stay unchanged. Run settings explicitly record the
optimization override. CUDA Graph timing claims cannot be inferred from this
ROCm runtime. The regression test ensures this happens before construction.

The owner environment is `${HOME}/.venvs/project3-greek-asr-nemo`, separate
from the old BANKING77 environment and from the Whisper/Qwen environment.
WSL's native Linux disk is used for package installation; the project's data,
code and checkpoints remain under the project directory. The initial NTFS
environment attempt was stopped because of installation overhead, not an
ASR compatibility result; its ignored partial files are not a runtime recipe.

From WSL, reproduce the exact environment with:

```bash
bash scripts/setup_nemo_rocm.sh
export HSA_ENABLE_DXG_DETECTION=1
"${HOME}/.venvs/project3-greek-asr-nemo/bin/python" -m greek_asr.infer --help
```

`P3_NEMO_ENV` can select a dedicated alternate environment path. The setup
script uses `requirements-nemo.lock.txt` when available. That file records
all installed versions and the official AMD torch, torchaudio and Triton
wheel URLs with hashes. It does not install into or change the older project's
environment. The observed stack is Python 3.12.3, torch
`2.9.1+rocm7.2.1.lw.gitff65f5bc` (distribution metadata), NeMo 3.0.0,
Transformers 4.57.6, NumPy 2.2.6, librosa 0.11.0 and soundfile 0.13.1.

The first actual import identified PyArrow 26's requirement for NumPy >=2.
The isolated environment was corrected from the initial NumPy 1.26.4 attempt
to 2.2.6, and the import then succeeded. `pip check` alone did not detect this
binary/import compatibility issue. No package was changed in the old
environment.

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
- [NeMo 3.0.0 greedy decoder CUDA Graph option](https://github.com/NVIDIA-NeMo/Speech/blob/v3.0.0/nemo/collections/asr/parts/submodules/rnnt_decoding.py)
- [Canary pinned source tree](https://huggingface.co/nvidia/canary-1b-v2/tree/d455706339a6b32e1aa40f82c713a482a0c938e2)
- [Parakeet pinned source tree](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/tree/541d1f99c6b0c3cd0b11a95167540bb8edefd82b)
