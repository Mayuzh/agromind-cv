#!/usr/bin/env bash
set -euo pipefail

VENV_PATH="${HOME}/.venvs/agromind-cv"
python3 -m venv "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-gpu.txt
python -m pip install -e .

echo
echo "Environment ready. Activate it with:"
echo "  source scripts/activate_wsl.sh"
source scripts/activate_wsl.sh
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"
