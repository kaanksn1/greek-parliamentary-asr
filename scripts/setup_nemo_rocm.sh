#!/usr/bin/env bash
# Isolated owner runtime for the pinned NeMo baselines on WSL/ROCm.
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
p3_nemo_env="${P3_NEMO_ENV:-${HOME}/.venvs/project3-greek-asr-nemo}"
amd_wheel_base="https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1"
lock_path="${project_root}/requirements-nemo.lock.txt"

if [[ ! -f "${p3_nemo_env}/pyvenv.cfg" ]]; then
    python3 -m venv "${p3_nemo_env}"
fi
p3_nemo_python="${p3_nemo_env}/bin/python"
"${p3_nemo_python}" -m pip install --disable-pip-version-check \
    'pip==26.0.1' 'setuptools==80.10.2' 'wheel==0.46.3'

if [[ -f "${lock_path}" ]]; then
    "${p3_nemo_python}" -m pip install --disable-pip-version-check -r "${lock_path}"
else
    "${p3_nemo_python}" -m pip install --disable-pip-version-check \
        "${amd_wheel_base}/torch-2.9.1%2Brocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl" \
        "${amd_wheel_base}/triton-3.5.1%2Brocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl" \
        "${amd_wheel_base}/torchaudio-2.9.0%2Brocm7.2.1.gite3c6ee2b-cp312-cp312-linux_x86_64.whl" \
        'nemo_toolkit[asr]==3.0.0' 'numpy==2.2.6' 'transformers==4.57.6' \
        'librosa==0.11.0' 'soundfile==0.13.1'
fi
"${p3_nemo_python}" -m pip check
"${p3_nemo_python}" -m pip freeze --all > "${lock_path}"
printf 'NeMo environment: %s\nExact lock: %s\n' "${p3_nemo_env}" "${lock_path}"
