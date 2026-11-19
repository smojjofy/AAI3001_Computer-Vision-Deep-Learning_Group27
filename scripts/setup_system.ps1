param(
    [ValidateSet("auto", "cpu", "cu126", "cu128")]
    [string]$TorchPlatform = "auto",

    [string]$Compiler = "",

    [switch]$RecreateVenv,

    [switch]$SkipNative,

    [switch]$SkipGui
)

$ErrorActionPreference = "Stop"




$repoRoot =
    [System.IO.Path]::GetFullPath(
        (Join-Path $PSScriptRoot "..")
    )

$venv =
    Join-Path $repoRoot ".venv"

$python =
    Join-Path $venv "Scripts\python.exe"

$buildNativeScript =
    Join-Path $repoRoot "scripts\build_splat_constructor.ps1"

$nativeBridge =
    Join-Path $repoRoot `
        "build\splat_constructor\splat_constructor_bridge.dll"

$nativeTests =
    Join-Path $repoRoot `
        "build\splat_constructor\splat_constructor_tests.exe"

$guiDir =
    Join-Path $repoRoot "applications\gui"

$guiDist =
    Join-Path $guiDir "dist\index.html"

$poseModel =
    Join-Path $repoRoot `
        "modules\inference\pose\models\pose_landmarker_lite.task"




function Write-Step {
    param(
        [string]$Message
    )

    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}


function Invoke-Checked {
    param(
        [string]$Executable,
        [string[]]$Arguments,
        [string]$WorkingDirectory = $repoRoot
    )

    Push-Location $WorkingDirectory

    try {
        & $Executable @Arguments

        if ($LASTEXITCODE -ne 0) {
            throw @"
Command failed with exit code $LASTEXITCODE

Executable:
$Executable

Arguments:
$($Arguments -join " ")
"@
        }
    }
    finally {
        Pop-Location
    }
}


function Find-CppCompiler {
    param(
        [string]$RequestedCompiler
    )

    if ($RequestedCompiler) {
        if (-not (
            Test-Path `
                -LiteralPath $RequestedCompiler
        )) {
            throw "Requested compiler not found: $RequestedCompiler"
        }

        return $RequestedCompiler
    }

    $candidates = @(
        "C:\msys64\ucrt64\bin\g++.exe",
        "C:\msys64\mingw64\bin\g++.exe"
    )

    foreach ($candidate in $candidates) {
        if (
            Test-Path `
                -LiteralPath $candidate
        ) {
            return $candidate
        }
    }

    $pathCompiler =
        Get-Command g++.exe `
            -ErrorAction SilentlyContinue

    if ($null -ne $pathCompiler) {
        return $pathCompiler.Source
    }

    return $null
}


function Find-NvidiaSmi {
    $command =
        Get-Command nvidia-smi.exe `
            -ErrorAction SilentlyContinue

    if ($null -ne $command) {
        return $command.Source
    }

    $candidate =
        Join-Path `
            $env:WINDIR `
            "System32\nvidia-smi.exe"

    if (
        Test-Path `
            -LiteralPath $candidate
    ) {
        return $candidate
    }

    $legacyCandidate =
        "C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"

    if (
        Test-Path `
            -LiteralPath $legacyCandidate
    ) {
        return $legacyCandidate
    }

    return $null
}


function Select-TorchPlatform {
    param(
        [string]$Requested
    )

    if ($Requested -ne "auto") {
        return $Requested
    }

    $nvidiaSmi =
        Find-NvidiaSmi

    if (-not $nvidiaSmi) {
        Write-Host `
            "No NVIDIA CUDA runtime detected. Using CPU PyTorch."

        return "cpu"
    }

    $report =
        & $nvidiaSmi 2>$null |
        Out-String

    if (
        $LASTEXITCODE -ne 0 -or
        $report -notmatch `
            "CUDA Version:\s*(\d+)\.(\d+)"
    ) {
        Write-Host `
            "NVIDIA detected, but CUDA capability could not be determined. Using CPU PyTorch."

        return "cpu"
    }

    $cudaVersion =
        [version]"$($Matches[1]).$($Matches[2])"

    Write-Host `
        "NVIDIA driver reports CUDA $cudaVersion"

    if (
        $cudaVersion -ge
        [version]"12.8"
    ) {
        return "cu128"
    }

    if (
        $cudaVersion -ge
        [version]"12.6"
    ) {
        return "cu126"
    }

    Write-Host `
        "CUDA is older than the supported project wheels. Falling back to CPU."

    return "cpu"
}




Write-Host ""
Write-Host "============================================"
Write-Host " Group 27 System Setup"
Write-Host "============================================"
Write-Host ""

Write-Host "Repository: $repoRoot"




Write-Step "Checking operating system"

if (
    [System.Environment]::OSVersion.Platform -ne
    [System.PlatformID]::Win32NT
) {
    throw @"
The current integrated build is Windows-specific.

The native Splat Constructor currently builds and loads:
splat_constructor_bridge.dll
"@
}

Write-Host "Windows detected."




Write-Step "Checking external prerequisites"




$pythonLauncher = $null
$pythonLauncherArgs = @()

$pyCommand =
    Get-Command py.exe `
        -ErrorAction SilentlyContinue

if ($null -ne $pyCommand) {
    $pythonLauncher =
        $pyCommand.Source

    $pythonLauncherArgs =
        @("-3")
}
else {
    $pythonCommand =
        Get-Command python.exe `
            -ErrorAction SilentlyContinue

    if ($null -ne $pythonCommand) {
        $pythonLauncher =
            $pythonCommand.Source
    }
}

if (-not $pythonLauncher) {
    throw @"
Python was not found.

Install Python 3.11 or newer, then run this script again.
"@
}




$npm =
    Get-Command npm.cmd `
        -ErrorAction SilentlyContinue

if (
    -not $SkipGui -and
    $null -eq $npm
) {
    throw @"
npm was not found.

Install Node.js, which includes npm, then run this script again.
"@
}



$compilerPath = $null

if (-not $SkipNative) {
    $compilerPath =
        Find-CppCompiler `
            -RequestedCompiler $Compiler

    if (-not $compilerPath) {
        throw @"
No supported C++ compiler was found.

Install MSYS2 with UCRT64 GCC.

Expected path:

C:\msys64\ucrt64\bin\g++.exe

The older MinGW64 path is also supported:

C:\msys64\mingw64\bin\g++.exe
"@
    }

    Write-Host "C++ compiler: $compilerPath"
}




Write-Step "Preparing Python virtual environment"

if ($RecreateVenv -and (Test-Path $venv)) {
    Write-Host "Removing existing .venv..."

    Remove-Item `
        -Recurse `
        -Force `
        $venv
}

if (-not (Test-Path $python)) {
    Write-Host "Creating .venv..."

    if ($pythonLauncherArgs.Count -gt 0) {
        & $pythonLauncher `
            @pythonLauncherArgs `
            -m venv `
            $venv
    }
    else {
        & $pythonLauncher `
            -m venv `
            $venv
    }

    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create .venv."
    }
}

if (-not (Test-Path $python)) {
    throw "Virtual environment Python was not created."
}




Write-Step "Checking Python version"

Invoke-Checked `
    -Executable $python `
    -Arguments @(
        "-c",
        "import sys; print('Python', sys.version.split()[0]); raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"
    )




Write-Step "Updating Python packaging tools"

Invoke-Checked `
    -Executable $python `
    -Arguments @(
        "-m",
        "pip",
        "install",
        "--upgrade",
        "pip",
        "setuptools",
        "wheel"
    )




Write-Step "Selecting PyTorch platform"

$selectedTorchPlatform =
    Select-TorchPlatform `
        -Requested $TorchPlatform

Write-Host `
    "PyTorch platform: $selectedTorchPlatform"

$torchIndex =
    "https://download.pytorch.org/whl/$selectedTorchPlatform"




Write-Step "Installing PyTorch"

Invoke-Checked `
    -Executable $python `
    -Arguments @(
        "-m",
        "pip",
        "install",
        "--upgrade",
        "torch==2.11.0",
        "torchvision==0.26.0",
        "--index-url",
        $torchIndex
    )




Write-Step "Installing integrated runtime packages"

Invoke-Checked `
    -Executable $python `
    -Arguments @(
        "-m",
        "pip",
        "install",
        "--upgrade",
        "ultralytics",
        "mediapipe",
        "numpy>=1.26,<3",
        "pytest"
    )




Write-Step "Configuring the single OpenCV installation"



& $python `
    -m pip `
    uninstall `
    -y `
    opencv-python `
    opencv-python-headless `
    opencv-contrib-python-headless



Invoke-Checked `
    -Executable $python `
    -Arguments @(
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--force-reinstall",
        "--no-deps",
        "opencv-contrib-python>=4.10,<5"
    )




Write-Step "Checking model assets"

if (-not (
    Test-Path `
        -LiteralPath $poseModel
)) {
    throw @"
MediaPipe Pose model is missing:

$poseModel
"@
}

Write-Host "Pose model OK."




Write-Step "Verifying Python runtime"

$verificationCode = @'
import torch
import cv2
import numpy
import mediapipe
import ultralytics

print("torch", torch.__version__)
print("CUDA available", torch.cuda.is_available())
print("cv2", cv2.__version__)
print("numpy", numpy.__version__)
print("mediapipe", mediapipe.__version__)
print("ultralytics", ultralytics.__version__)
'@

$verificationCode |
    & $python -

if ($LASTEXITCODE -ne 0) {
    throw "Python runtime verification failed."
}



if (-not $SkipNative) {
    Write-Step "Building native Splat Constructor"

    & $buildNativeScript `
        -Compiler $compilerPath

    if ($LASTEXITCODE -ne 0) {
        throw "Native Splat Constructor build failed."
    }

    if (-not (
        Test-Path `
            -LiteralPath $nativeBridge
    )) {
        throw @"
Native bridge was not produced:

$nativeBridge
"@
    }

    if (-not (
        Test-Path `
            -LiteralPath $nativeTests
    )) {
        throw @"
Native test executable was not produced:

$nativeTests
"@
    }

    Write-Host "Running native tests..."

    & $nativeTests

    if ($LASTEXITCODE -ne 0) {
        throw "Native Splat Constructor tests failed."
    }

    Write-Host "Native tests passed."




    Write-Host "Checking Python/native bridge..."

    Invoke-Checked `
        -Executable $python `
        -Arguments @(
            "-c",
            "import ctypes; from modules.splat_constructor.python_bridge import default_library_path; p=default_library_path(); ctypes.CDLL(str(p)); print('Native bridge OK:', p)"
        )
}




if (-not $SkipGui) {
    Write-Step "Installing and building GUI"

    Invoke-Checked `
        -Executable $npm.Source `
        -Arguments @(
            "ci"
        ) `
        -WorkingDirectory $guiDir

    Invoke-Checked `
        -Executable $npm.Source `
        -Arguments @(
            "run",
            "build"
        ) `
        -WorkingDirectory $guiDir

    if (-not (
        Test-Path `
            -LiteralPath $guiDist
    )) {
        throw @"
GUI build did not produce:

$guiDist
"@
    }

    Write-Host "GUI build OK."
}




Write-Step "Checking integrated application"

Invoke-Checked `
    -Executable $python `
    -Arguments @(
        "scripts\run_integration_app.py",
        "--help"
    ) `
    -WorkingDirectory $repoRoot




Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host " Setup complete" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""

Write-Host "PyTorch platform : $selectedTorchPlatform"

if (-not $SkipNative) {
    Write-Host "Native bridge    : ready"
}

if (-not $SkipGui) {
    Write-Host "GUI              : ready"
}

Write-Host ""
Write-Host "Start the complete system with:"
Write-Host ""
Write-Host "  .\scripts\start_system.ps1" -ForegroundColor Cyan
Write-Host ""