#!/usr/bin/env bash
set -euo pipefail
source scripts/activate_wsl.sh
IMAGE_PATH="${1:?Pass an image path as the first argument}"
python src/predict.py --image "${IMAGE_PATH}"
python scripts/smoke_api.py "${IMAGE_PATH}"
