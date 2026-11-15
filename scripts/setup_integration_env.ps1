# Create the integration Python environment.
#
# The system Python has a CPU-only torch wheel. This script creates a local
# .venv with a CUDA torch build for the RTX 4050 and installs the runtime
# dependencies of the integrated pipeline.
#
# OpenCV: ultralytics declares opencv-python, and the pose module declares
# opencv-contrib-python. Only one wheel flavor may own cv2. Install ultralytics
# first, remove the plain wheel, then force a clean contrib install. OpenCV is
# pinned below 5 because the frozen modules use the 4.x API.

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $root '.venv'
$py = Join-Path $venv 'Scripts\python.exe'

if (-not (Test-Path $py)) {
    Write-Host "Creating venv at $venv"
    python -m venv $venv
}

Write-Host 'Upgrading pip'
& $py -m pip install --upgrade pip setuptools wheel

Write-Host 'Installing CUDA torch for the RTX 4050'
& $py -m pip install 'torch==2.11.0+cu128' torchvision --index-url https://download.pytorch.org/whl/cu128

Write-Host 'Installing runtime packages'
& $py -m pip install ultralytics mediapipe numpy pytest

Write-Host 'Removing the conflicting plain OpenCV wheel if present'
& $py -m pip uninstall -y opencv-python

Write-Host 'Installing one clean OpenCV 4.x wheel (contrib)'
& $py -m pip install --force-reinstall --no-deps 'opencv-contrib-python>=4.10,<5'

Write-Host 'Verifying'
& $py -c "import torch, cv2, numpy; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available()); print('cv2', cv2.__version__, 'numpy', numpy.__version__)"
& $py -c "import mediapipe, ultralytics; print('mediapipe', mediapipe.__version__); print('ultralytics', ultralytics.__version__)"
Write-Host 'Environment ready.'
