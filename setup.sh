#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-shot environment setup for Linux / WSL2.
# Creates .venv, installs the Blackwell-compatible PyTorch, then the rest.
# Usage:  bash setup.sh
# ---------------------------------------------------------------------------
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV_DIR=".venv"
# CUDA 12.8 wheels are the ones built with sm_120 kernels (RTX 50-series).
TORCH_INDEX="https://download.pytorch.org/whl/cu128"

echo "==> Creating virtual environment with ${PYTHON_BIN}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "==> Upgrading pip toolchain"
python -m pip install --upgrade pip setuptools wheel

echo "==> Installing PyTorch (CUDA 12.8 build, required for RTX 5080 / sm_120)"
pip install torch torchvision torchaudio --index-url "${TORCH_INDEX}"

echo "==> Installing project dependencies"
pip install -r requirements.txt

if [ ! -f .env ]; then
  echo "==> Creating .env from the template (fill in your HF_TOKEN)"
  cp .env.example .env
  chmod 600 .env          # owner-only: it will hold a secret
fi

echo "==> Verifying the GPU is actually usable"
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available - check your NVIDIA driver."
cap = torch.cuda.get_device_capability(0)
print(f"    GPU        : {torch.cuda.get_device_name(0)}")
print(f"    Capability : sm_{cap[0]}{cap[1]}")
print(f"    Torch/CUDA : {torch.__version__} / {torch.version.cuda}")
if cap[0] >= 12:
    # A wheel without sm_120 kernels fails exactly here, not at import time.
    torch.zeros(8, device="cuda") @ torch.zeros(8, 8, device="cuda")
    print("    sm_120 matmul smoke test: OK")
PY

echo
echo "Done. Next steps:"
echo "  1. source ${VENV_DIR}/bin/activate"
echo "  2. edit .env and set HF_TOKEN (https://huggingface.co/settings/tokens)"
echo "  3. python scripts/00_record_env.py"
