# Thin wrapper kept for existing documentation and habits.
#
# The canonical setup script is scripts/setup_system.ps1: it checks the operating
# system and external prerequisites, creates the virtual environment, installs
# PyTorch from a CUDA wheel index, installs the runtime packages, handles the
# single-OpenCV rule, builds the GUI, builds and tests the native Splat
# Constructor, and runs the Python test suite.
#
# This file forwards every argument to that script.
#
#   .\scripts\setup_device.ps1
#   .\scripts\setup_device.ps1 -SkipNative
#   .\scripts\setup_device.ps1 -WithRaft          # accepted for compatibility
#
# -WithRaft and -TorchWheel are accepted so older commands keep working:
# -TorchWheel maps to -TorchPlatform, and -WithRaft is ignored because the
# canonical script always installs PyTorch.
param(
    [switch]$WithRaft,
    [ValidateSet('auto', 'cpu', 'cu118', 'cu126', 'cu128')]
    [string]$TorchWheel = 'auto',
    [switch]$SkipNative,
    [switch]$SkipGui,
    [switch]$SkipTests,
    [switch]$RecreateVenv
)

$ErrorActionPreference = 'Stop'

$canonical = Join-Path $PSScriptRoot 'setup_system.ps1'
if (-not (Test-Path -LiteralPath $canonical)) {
    throw "Canonical setup script not found: $canonical"
}

$forward = @{}
switch ($TorchWheel) {
    'cpu' { $forward['TorchPlatform'] = 'cpu' }
    'cu118' { $forward['TorchPlatform'] = 'cu126' }
    'cu126' { $forward['TorchPlatform'] = 'cu126' }
    'cu128' { $forward['TorchPlatform'] = 'cu128' }
    default { $forward['TorchPlatform'] = 'auto' }
}
if ($SkipNative) { $forward['SkipNative'] = $true }
if ($SkipGui) { $forward['SkipGui'] = $true }
if ($SkipTests) { $forward['SkipTests'] = $true }
if ($RecreateVenv) { $forward['RecreateVenv'] = $true }

& $canonical @forward
exit $LASTEXITCODE
