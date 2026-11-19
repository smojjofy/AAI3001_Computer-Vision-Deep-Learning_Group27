param(
    [ValidateSet("webcam", "camera", "obs", "video")]
    [string]$Source = "webcam",

    [int]$Device = 0,

    [ValidateSet("auto", "dshow", "msmf")]
    [string]$CameraBackend = "dshow",

    [string]$VideoPath = "",

    [ValidateSet("auto", "cpu", "cuda")]
    [string]$InferenceDevice = "auto",

    [int]$Port = 8765,

    [string]$Compiler = "",

    [switch]$RebuildNative,

    [switch]$RebuildGui,

    [switch]$NoBrowser,

    [string[]]$ExtraAppArgs = @()
)

$ErrorActionPreference = "Stop"




$repoRoot =
    [System.IO.Path]::GetFullPath(
        (Join-Path $PSScriptRoot "..")
    )

$python =
    Join-Path $repoRoot ".venv\Scripts\python.exe"

$appScript =
    Join-Path $repoRoot "scripts\run_integration_app.py"

$buildNativeScript =
    Join-Path $repoRoot "scripts\build_splat_constructor.ps1"

$nativeBuildDir =
    Join-Path $repoRoot "build\splat_constructor"

$nativeBridge =
    Join-Path $nativeBuildDir "splat_constructor_bridge.dll"

$nativeTests =
    Join-Path $nativeBuildDir "splat_constructor_tests.exe"

$guiDir =
    Join-Path $repoRoot "applications\gui"

$guiDist =
    Join-Path $guiDir "dist\index.html"

$nodeModules =
    Join-Path $guiDir "node_modules"

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
            throw "Command failed with exit code $LASTEXITCODE`: $Executable $($Arguments -join ' ')"
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
        if (-not (Test-Path -LiteralPath $RequestedCompiler)) {
            throw "Requested C++ compiler not found: $RequestedCompiler"
        }

        return $RequestedCompiler
    }

    $candidates = @(
        "C:\msys64\ucrt64\bin\g++.exe",
        "C:\msys64\mingw64\bin\g++.exe"
    )

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
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




Write-Host ""
Write-Host "============================================"
Write-Host " Group 27 Live Splat Reconstruction"
Write-Host "============================================"
Write-Host ""

Write-Host "Repository : $repoRoot"
Write-Host "Source     : $Source"
Write-Host "Device     : $Device"




Write-Step "Checking Python environment"

if (-not (Test-Path -LiteralPath $python)) {
    throw @"
Python virtual environment was not found:

$python

Prepare the machine's Python environment before running the launcher.
"@
}

if (-not (Test-Path -LiteralPath $appScript)) {
    throw "Integrated application not found: $appScript"
}

if (-not (Test-Path -LiteralPath $poseModel)) {
    throw @"
MediaPipe Pose model was not found:

$poseModel
"@
}

Invoke-Checked `
    -Executable $python `
    -Arguments @(
        "-c",
        "import torch, cv2, numpy, mediapipe, ultralytics; print('Python runtime OK'); print('torch', torch.__version__); print('cv2', cv2.__version__)"
    )



Write-Step "Selecting inference device"

$cudaResult =
    & $python -c "import torch; print('true' if torch.cuda.is_available() else 'false')"

if ($LASTEXITCODE -ne 0) {
    throw "Unable to query PyTorch device availability."
}

$cudaAvailable =
    (($cudaResult | Out-String).Trim() -eq "true")

if ($InferenceDevice -eq "auto") {
    if ($cudaAvailable) {
        $selectedInferenceDevice = "cuda"
    }
    else {
        $selectedInferenceDevice = "cpu"
    }
}
elseif ($InferenceDevice -eq "cuda") {
    if (-not $cudaAvailable) {
        throw "CUDA was requested, but PyTorch reports that CUDA is unavailable."
    }

    $selectedInferenceDevice = "cuda"
}
else {
    $selectedInferenceDevice = "cpu"
}

Write-Host "Inference device: $selectedInferenceDevice"

if ($selectedInferenceDevice -eq "cpu") {
    Write-Host "CUDA unavailable or disabled. YOLO depth will run on CPU."
}


$nativeNeedsBuild =
    $RebuildNative -or
    (-not (Test-Path -LiteralPath $nativeBridge))

if ($nativeNeedsBuild) {
    Write-Step "Building native Splat Constructor"

    $compilerPath =
        Find-CppCompiler `
            -RequestedCompiler $Compiler

    if (-not $compilerPath) {
        throw @"
No suitable C++ compiler was found.

Expected one of:

C:\msys64\ucrt64\bin\g++.exe
C:\msys64\mingw64\bin\g++.exe

You may also supply one explicitly:

.\scripts\start_system.ps1 -Compiler "C:\path\to\g++.exe"
"@
    }

    Write-Host "Compiler: $compilerPath"

    & $buildNativeScript `
        -Compiler $compilerPath

    if (-not (Test-Path -LiteralPath $nativeBridge)) {
        throw "Native bridge was not produced: $nativeBridge"
    }

    if (Test-Path -LiteralPath $nativeTests) {
        Write-Host "Running native constructor tests..."

        & $nativeTests

        if ($LASTEXITCODE -ne 0) {
            throw "Native Splat Constructor tests failed."
        }
    }
}
else {
    Write-Step "Native Splat Constructor already built"
    Write-Host $nativeBridge
}



Write-Step "Checking native bridge"

Invoke-Checked `
    -Executable $python `
    -Arguments @(
        "-c",
        "import ctypes; from modules.splat_constructor.python_bridge import default_library_path; p=default_library_path(); ctypes.CDLL(str(p)); print('Native bridge OK:', p)"
    )


$guiNeedsBuild =
    $RebuildGui -or
    (-not (Test-Path -LiteralPath $guiDist))

if ($guiNeedsBuild) {
    Write-Step "Building GUI"

    $npm =
        Get-Command npm.cmd `
            -ErrorAction SilentlyContinue

    if ($null -eq $npm) {
        throw "npm was not found. Install Node.js before building the GUI."
    }

    if (-not (Test-Path -LiteralPath $nodeModules)) {
        Write-Host "GUI dependencies missing. Running npm ci..."

        Invoke-Checked `
            -Executable $npm.Source `
            -Arguments @("ci") `
            -WorkingDirectory $guiDir
    }

    Invoke-Checked `
        -Executable $npm.Source `
        -Arguments @("run", "build") `
        -WorkingDirectory $guiDir

    if (-not (Test-Path -LiteralPath $guiDist)) {
        throw "GUI build did not produce: $guiDist"
    }
}
else {
    Write-Step "GUI already built"
    Write-Host $guiDist
}


if ($Source -eq "video") {
    if (-not $VideoPath) {
        throw "-VideoPath is required when -Source video is used."
    }

    if (-not (Test-Path -LiteralPath $VideoPath)) {
        throw "Video file not found: $VideoPath"
    }
}


Write-Step "Checking port $Port"

$portInUse = $false

if (Get-Command Get-NetTCPConnection -ErrorAction SilentlyContinue) {
    $listener =
        Get-NetTCPConnection `
            -State Listen `
            -LocalPort $Port `
            -ErrorAction SilentlyContinue

    $portInUse =
        $null -ne $listener
}

if ($portInUse) {
    throw "Port $Port is already in use. Stop the existing process or choose another port."
}

$appArguments = @(
    $appScript,
    "--source", $Source,
    "--device", "$Device",
    "--camera-backend", $CameraBackend,
    "--port", "$Port",
    "--device-inference", $selectedInferenceDevice
)

if ($Source -eq "video") {
    $appArguments += @(
        "--path",
        $VideoPath
    )
}

if ($ExtraAppArgs.Count -gt 0) {
    $appArguments +=
        $ExtraAppArgs
}


Write-Step "Starting integrated application"

Write-Host ""
Write-Host "Command:"
Write-Host "$python $($appArguments -join ' ')"
Write-Host ""

$quotedArguments =
    $appArguments |
    ForEach-Object {
        if ($_ -match "\s") {
            '"' +
            ($_ -replace '"', '\"') +
            '"'
        }
        else {
            $_
        }
    }

$argumentLine =
    $quotedArguments -join " "

$process =
    Start-Process `
        -FilePath $python `
        -ArgumentList $argumentLine `
        -WorkingDirectory $repoRoot `
        -NoNewWindow `
        -PassThru




$healthUrl =
    "http://127.0.0.1:$Port/api/health"

$appUrl =
    "http://127.0.0.1:$Port/"

Write-Step "Waiting for server"

$ready = $false

for ($attempt = 1; $attempt -le 60; $attempt++) {
    if ($process.HasExited) {
        throw "Integrated application exited before the server became ready."
    }

    try {
        $response =
            Invoke-WebRequest `
                -Uri $healthUrl `
                -UseBasicParsing `
                -TimeoutSec 1

        if ($response.StatusCode -eq 200) {
            $ready = $true
            break
        }
    }
    catch {
        
    }

    Start-Sleep -Seconds 1
}

if (-not $ready) {
    Write-Warning "The server did not answer /api/health within 60 seconds."
}
else {
    Write-Host "Server ready: $healthUrl"
}




if (-not $NoBrowser) {
    Write-Step "Opening GUI"
    Start-Process $appUrl
}




Write-Host ""
Write-Host "============================================"
Write-Host " System running"
Write-Host "============================================"
Write-Host ""
Write-Host "GUI              : $appUrl"
Write-Host "Inference device : $selectedInferenceDevice"
Write-Host "Source           : $Source"
Write-Host ""
Write-Host "Press Ctrl+C to stop."
Write-Host ""

try {
    Wait-Process `
        -Id $process.Id
}
finally {
    if (-not $process.HasExited) {
        Write-Host ""
        Write-Host "Stopping integrated application..."

        Stop-Process `
            -Id $process.Id `
            -Force `
            -ErrorAction SilentlyContinue
    }
}