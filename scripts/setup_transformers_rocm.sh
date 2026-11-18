#!/usr/bin/env bash
# Separate Python 3.12 runtime for the owner's standard Whisper/Qwen baselines.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
if [[ ! -d "$PROJECT_ROOT/greek_asr" || ! -f "$PROJECT_ROOT/configs/model_sources.lock.json" ]]; then
  echo "Unexpected project directory; refusing environment setup." >&2
  exit 1
fi
P3_ENV="${P3_TRANSFORMERS_ENV:-$HOME/.venvs/project3-greek-asr-transformers}"
if [[ "${1:-}" == "--env" && -n "${2:-}" ]]; then
  P3_ENV="$2"
  shift 2
fi
if [[ "$#" -ne 0 ]]; then
  echo "Usage: bash scripts/setup_transformers_rocm.sh [--env /absolute/path]" >&2
  exit 1
fi
python3.12 -m venv "$P3_ENV"
P3_PYTHON="$P3_ENV/bin/python"
mkdir -p outputs
if [[ -f requirements-rocm.lock.txt ]]; then
  "$P3_PYTHON" -m pip install --find-links https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/ -r requirements-rocm.lock.txt
else
  "$P3_PYTHON" -m pip install 'pip==26.3'
  "$P3_PYTHON" -m pip install --find-links https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/ \
    'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl' \
    'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torchvision-0.24.0%2Brocm7.2.1.gitb919bd0c-cp312-cp312-linux_x86_64.whl' \
    'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torchaudio-2.9.0%2Brocm7.2.1.gite3c6ee2b-cp312-cp312-linux_x86_64.whl'
  printf '%s\n' 'torch==2.9.1+rocm7.2.1.lw.gitff65f5bc' \
    'torchvision==0.24.0+rocm7.2.1.gitb919bd0c' \
    'torchaudio==2.9.0+rocm7.2.1.gite3c6ee2b' \
    'triton==3.5.1+rocm7.2.1.gita272dfa8' > outputs/transformers-bootstrap-constraints.txt
  "$P3_PYTHON" -m pip install -c outputs/transformers-bootstrap-constraints.txt \
    'numpy==1.26.4' 'transformers==4.57.6' 'accelerate==1.12.0' \
    'qwen-asr==0.0.6' 'setuptools==80.9.0'
  # Includes exact transitive packages and the official AMD wheel URL.
  "$P3_PYTHON" -m pip freeze --all > requirements-rocm.lock.txt
fi
"$P3_PYTHON" -m pip check
HSA_ENABLE_DXG_DETECTION=1 "$P3_PYTHON" -c \
  'import torch; print({"torch":torch.__version__,"hip":torch.version.hip,"gpu_available":torch.cuda.is_available()}); assert torch.cuda.is_available(), "CUDA/ROCm device unavailable"'
