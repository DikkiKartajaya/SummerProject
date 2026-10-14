# ---------------------------------------------------------------------------
# One-shot environment setup for Windows (PowerShell).
# Usage:  powershell -ExecutionPolicy Bypass -File setup.ps1
# ---------------------------------------------------------------------------
$ErrorActionPreference = "Stop"

$VenvDir    = ".venv"
# CUDA 12.8 wheels are the ones built with sm_120 kernels (RTX 50-series).
$TorchIndex = "https://download.pytorch.org/whl/cu128"

Write-Host "==> Creating virtual environment"
py -3.11 -m venv $VenvDir

Write-Host "==> Activating"
& "$VenvDir\Scripts\Activate.ps1"

Write-Host "==> Upgrading pip toolchain"
python -m pip install --upgrade pip setuptools wheel

Write-Host "==> Installing PyTorch (CUDA 12.8 build, required for RTX 5080 / sm_120)"
pip install torch torchvision torchaudio --index-url $TorchIndex

Write-Host "==> Installing project dependencies"
pip install -r requirements.txt

if (-not (Test-Path ".env")) {
    Write-Host "==> Creating .env from the template (fill in your HF_TOKEN)"
    Copy-Item ".env.example" ".env"
}

Write-Host "==> Verifying the GPU is actually usable"
python -c @"
import torch
assert torch.cuda.is_available(), 'CUDA not available - check your NVIDIA driver.'
cap = torch.cuda.get_device_capability(0)
print('    GPU        :', torch.cuda.get_device_name(0))
print('    Capability : sm_%d%d' % cap)
print('    Torch/CUDA :', torch.__version__, '/', torch.version.cuda)
if cap[0] >= 12:
    torch.zeros(8, device='cuda') @ torch.zeros(8, 8, device='cuda')
    print('    sm_120 matmul smoke test: OK')
"@

Write-Host ""
Write-Host "Done. Next steps:"
Write-Host "  1. .\$VenvDir\Scripts\Activate.ps1"
Write-Host "  2. edit .env and set HF_TOKEN (https://huggingface.co/settings/tokens)"
Write-Host "  3. python scripts/00_record_env.py"
