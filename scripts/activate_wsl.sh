#!/usr/bin/env bash
# Source this file; do not execute it: source scripts/activate_wsl.sh
source "${HOME}/.venvs/agromind-cv/bin/activate"
NVIDIA_ROOT="${VIRTUAL_ENV}/lib/python3.12/site-packages/nvidia"
if [[ -d "${NVIDIA_ROOT}" ]]; then
  NVIDIA_LIB_DIRS="$(find "${NVIDIA_ROOT}" -type d -name lib -print | paste -sd: -)"
  export LD_LIBRARY_PATH="${NVIDIA_LIB_DIRS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  export PATH="${NVIDIA_ROOT}/cuda_nvcc/bin:${PATH}"
  export XLA_FLAGS="--xla_gpu_cuda_data_dir=${NVIDIA_ROOT}/cuda_nvcc"
fi

