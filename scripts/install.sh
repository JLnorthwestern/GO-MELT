#!/usr/bin/env bash
set -euo pipefail
# Usage: ./scripts/install.sh [--force-gpu] [venv-path]
# Example: ./scripts/install.sh --force-gpu .venv

VENV_PATH="${2:-${1:-.venv}}"
FORCE_GPU=false
if [ "${1:-}" = "--force-gpu" ] || [ "${2:-}" = "--force-gpu" ]; then
  FORCE_GPU=true
fi

echo "Creating venv at: $VENV_PATH"
python3 -m venv "$VENV_PATH"
# shellcheck disable=SC1090
source "$VENV_PATH/bin/activate"

python -m pip install --upgrade pip setuptools wheel

detect_gpu() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi
  return 1
}

if [ "$FORCE_GPU" = true ] || detect_gpu; then
  echo "GPU detected or forced. Installing GPU requirements..."
  pip install --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html -r requirements-gpu.txt
  pip install -e .
else
  echo "No GPU detected. Installing CPU/dev requirements..."
  pip install -r requirements.txt
  pip install -e .
fi

echo "Installation complete. Activate the venv with: source $VENV_PATH/bin/activate"

