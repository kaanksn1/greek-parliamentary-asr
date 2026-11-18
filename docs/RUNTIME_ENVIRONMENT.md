# Isolated Whisper / Qwen ROCm runtime

Prepared and exercised on 18 November 2026. This is the owner's inference
environment; it does not install teammates' training implementations or the
separate NeMo backend environment. The old BANKING77/RAGTruth runtime at
`/opt/banking77-venv` was not changed.

## Platform and exact dependencies

Actual platform: WSL Ubuntu 24.04, Python 3.12.3, AMD Radeon RX 7800 XT.
The successful isolated environment is
`/home/serda/.venvs/project3-greek-asr-transformers`. WSL's native Linux
filesystem was selected after an initial NTFS environment proved slow.
The project's ignored partial `.venv-rocm` is not the working runtime.

[requirements-rocm.lock.txt](../requirements-rocm.lock.txt) is the actual
`pip freeze --all` result, including exact transitive dependencies. Key
components are Transformers 4.57.6, Qwen-ASR 0.0.6, Accelerate 1.12.0 and
NumPy 1.26.4. Torch is the AMD 2.9.1 ROCm 7.2.1 Linux/Python 3.12 wheel;
the lock retains its download URL and SHA-256. Its matching AMD Triton,
torchvision and torchaudio distributions are recorded as well. The observed
Torch build string and HIP version are in each successful run's environment
record; package metadata and Torch's display version can differ.

The initial Torch-only install could not resolve its AMD-specific Triton
version from PyPI. The setup script now supplies the official
[AMD 7.2.1 wheel index](https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/)
for installation and lock replay. It does not substitute CUDA wheels.
`pip check` reported no broken requirements after installation.

This lock targets the tested Linux x86-64 / CPython 3.12 / ROCm platform.
It is not a Windows CPU environment or a GPU-driver installation recipe.
The existing compatible WSL GPU driver/runtime is a prerequisite.

## Setup and reuse

From the cloned repository's root in WSL:

```bash
bash scripts/setup_transformers_rocm.sh
```

The default is `$HOME/.venvs/project3-greek-asr-transformers`; an explicit
separate destination is supported:

```bash
bash scripts/setup_transformers_rocm.sh --env "$HOME/.venvs/project3-greek-asr-transformers"
```

When the lock exists, the script installs from that exact lock. The initial
bootstrap recipe is retained for provenance; it resolves and writes the lock
only when no lock is present. It then checks dependencies and GPU visibility.
The script identifies a checkout by its files rather than the directory name,
so the public repository's ordinary clone name is supported.

For this WSL installation, **`HSA_ENABLE_DXG_DETECTION=1` is required** when
running Python with GPU access. The installation probe and actual runs used
that value. A requested unavailable GPU raises an error rather than silently
running on CPU. Qwen retains upstream SDPA submodules; experimental AMD
attention warnings were observed, but both sample runs completed. No
FlashAttention 2 package or experimental AOTriton flag was enabled.

## Reproducible train-only runtime checks

The common preparation script audits the three existing original HParl
training WAVs and writes an audio-only manifest. It does not prepare the full
dataset or freeze the official test; that remains Samed's assigned work.

```bash
P3_PYTHON="$HOME/.venvs/project3-greek-asr-transformers/bin/python"
"$P3_PYTHON" scripts/prepare_owner_pilot.py
HSA_ENABLE_DXG_DETECTION=1 "$P3_PYTHON" -m greek_asr.infer \
  --manifest outputs/owner_pilot_manifest.jsonl --backend whisper --split train \
  --purpose runtime-smoke --output outputs/repeat_whisper_train
HSA_ENABLE_DXG_DETECTION=1 "$P3_PYTHON" -m greek_asr.infer \
  --manifest outputs/owner_pilot_manifest.jsonl --backend qwen --split train \
  --purpose runtime-smoke --output outputs/repeat_qwen_train
```

Each output directory must be new; the runner preserves existing results.
Checkpoints use exact revisions from `configs/model_sources.lock.json`,
including on cache reuse. Model caches are excluded from Git. Download/load
time is separate from the recorded inference interval; the latter includes
output I/O and is not a controlled warm-throughput measurement.

Committed sample evidence is under:

- `results/runtime_smoke/owner_whisper_train/`
- `results/runtime_smoke/owner_qwen_train_verified/`

Both contain predictions for all three original training clips and an actual
`run.json` recording source/audio identity, model revision, settings, code
hashes, environment, timing and peak GPU allocation. The corrected Qwen run
records the requested top-level eager attention separately from the observed
SDPA submodules. Its earlier local run used an oversimplified attention label;
the verified record is the evidence referenced here. Original predictions and
source audio were not edited. Changes to unrelated adapter files after a run
can change the runner's broad code-hash inventory; historical hashes refer to
the exact source state used for that run, not a claim that every later file
is byte-identical.

These checks establish runtime compatibility on a tiny training sample.
They perform no model training, no WER/CER scoring, and no official dev/test
inference. Full-data quality, feasible training budget and final comparison
remain separate tasks.
