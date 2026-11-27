# Start the live splat pipeline and open the browser viewer.
#
# Run from anywhere:
#   .\start_live_demo.ps1                      # webcam, the default
#   .\start_live_demo.ps1 -Source zed          # ZED 2i on USB
#   .\start_live_demo.ps1 -Source auto         # webcam, or ZED when it is connected
#   .\start_live_demo.ps1 -DurationS 180 -Port 8765
#   .\start_live_demo.ps1 -Source video -VideoPath .tmp\clip.mp4
#   .\start_live_demo.ps1 -ListZed             # list connected ZED cameras and exit
#
# The script checks the environment, starts the integrated app in a new window,
# waits until a splat state is published, then opens the viewer in your browser.
# Press Ctrl+C in the app window to stop it, or run with -DurationS to stop it
# automatically.
param(
    [ValidateSet('auto', 'camera', 'webcam', 'obs', 'zed', 'video')]
    [string]$Source = 'webcam',
    [int]$Device = 0,
    [switch]$ListZed,
    [ValidateSet('HD720', 'HD1080', 'HD2K', 'VGA')]
    [string]$ZedResolution = 'HD720',
    [int]$ZedFps = 30,
    [ValidateSet('NEURAL', 'NEURAL_PLUS', 'ULTRA', 'QUALITY', 'PERFORMANCE')]
    [string]$ZedDepthMode = 'QUALITY',
    [ValidateSet('auto', 'dshow', 'msmf')]
    [string]$CameraBackend = 'dshow',
    [int]$Port = 8765,
    [double]$DurationS = 0.0,
    [string]$VideoPath = '',
    [double]$Threshold = 0.002,
    [int]$TimeoutSec = 120,
    [switch]$RebuildGui,
    [switch]$NoBrowser,
    # Shadow mask run: compute a research mask candidate beside each refresh and
    # record it. The published splat does not change.
    [string]$ShadowMaskCheckpoint = '',
    [string]$ShadowMaskLog = '',
    [switch]$FaceDepthCorrection,
    [string]$FaceDepthRuleJson = '',
    [string]$FaceDepthLog = ''
)

$ErrorActionPreference = 'Stop'

# The component repository is the folder that contains this script.
$repoRoot = $PSScriptRoot
if (-not (Test-Path -LiteralPath (Join-Path $repoRoot 'scripts\run_integration_app.py'))) {
    throw "This script must sit in the live-splat-pipeline folder. Not found: scripts\run_integration_app.py"
}

$python = Join-Path $repoRoot '.venv\Scripts\python.exe'
$appScript = Join-Path $repoRoot 'scripts\run_integration_app.py'
$guiDist = Join-Path $repoRoot 'applications\gui\dist\index.html'
$guiRoot = Join-Path $repoRoot 'applications\gui'
$logDir = Join-Path $repoRoot '.tmp\demo'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$outLog = Join-Path $logDir "run-$stamp.log"

function Write-Step {
    param([string]$Message)
    Write-Host "[start] $Message" -ForegroundColor Cyan
}

# ---------------------------------------------------------------- environment

Write-Step 'Checking the Python virtual environment'
if (-not (Test-Path -LiteralPath $python)) {
    throw "Python venv not found at $python. Run .\scripts\setup_system.ps1 first."
}

$env:PYTHONNOUSERSITE = '1'
$tmpDir = Join-Path $repoRoot '.tmp'
New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null
$env:TMP = (Resolve-Path $tmpDir).Path
$env:TEMP = $env:TMP

Write-Step 'Checking the installed modules'
& $python -c "import cv2, stream_handler, optical_flow, reranker" 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Python modules are missing. Run .\scripts\setup_system.ps1, then run this script again."
}

Write-Step 'Checking the native Splat Constructor'
$nativeBridge = Join-Path $repoRoot 'build\splat_constructor\splat_constructor_bridge.dll'
if (-not (Test-Path -LiteralPath $nativeBridge)) {
    Write-Warning "Native bridge not found at build\splat_constructor\splat_constructor_bridge.dll. Run .\scripts\build_splat_constructor.ps1 if the pipeline cannot build splats."
}

Write-Step 'Checking the GUI build'
$needBuild = $RebuildGui -or (-not (Test-Path -LiteralPath $guiDist))
if (-not $needBuild) {
    $guiSource = Get-ChildItem -Path (Join-Path $guiRoot 'src') -Recurse -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -ne $guiSource -and $guiSource.LastWriteTime -gt (Get-Item -LiteralPath $guiDist).LastWriteTime) {
        Write-Warning 'GUI sources are newer than the build; rebuilding.'
        $needBuild = $true
    }
}
if ($needBuild) {
    if (-not (Test-Path -LiteralPath (Join-Path $guiRoot 'node_modules'))) {
        throw "GUI dependencies are missing. Run 'npm install' in applications\gui first."
    }
    Write-Step 'Building the GUI (this takes a few seconds)'
    Push-Location $guiRoot
    try {
        & npm run build 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "GUI build failed. Run 'npm run build' in applications\gui to see the error."
        }
    }
    finally {
        Pop-Location
    }
}

# ------------------------------------------------------------------ the app

if ($ListZed) {
    # Show the connected ZED cameras and stop. This needs no GUI and no server.
    & $python -m scripts.run_integration_app --list-zed
    exit $LASTEXITCODE
}

if ($Source -eq 'video') {
    if ([string]::IsNullOrWhiteSpace($VideoPath)) {
        throw "Pass -VideoPath when -Source is 'video'."
    }
    if (-not (Test-Path -LiteralPath $VideoPath)) {
        throw "Video file not found: $VideoPath"
    }
    $resolvedVideo = (Resolve-Path $VideoPath).Path
}

# Refuse to start on a port that is already serving.
$existing = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($null -ne $existing) {
    throw "Port $Port is already in use. Close the other pipeline window, or pass -Port with another number."
}

$appArgs = @(
    $appScript,
    '--source', $Source,
    '--port', "$Port",
    '--threshold', "$Threshold"
)
switch ($Source) {
    'camera' {
        $appArgs += @('--device', "$Device", '--camera-backend', $CameraBackend)
    }
    'webcam' {
        $appArgs += @('--device', "$Device", '--camera-backend', $CameraBackend)
    }
    'zed' {
        # The ZED SDK reports its own camera index, so -Device is not forwarded.
        $appArgs += @('--zed-camera-id', "$Device", '--zed-resolution', $ZedResolution, '--zed-fps', "$ZedFps", '--zed-depth-mode', $ZedDepthMode)
    }
    'auto' {
        # --source auto decides inside the app: webcam first, ZED when present.
        # The webcam arguments stay valid for the fallback.
        $appArgs += @('--device', "$Device", '--camera-backend', $CameraBackend)
        $appArgs += @('--zed-resolution', $ZedResolution, '--zed-fps', "$ZedFps", '--zed-depth-mode', $ZedDepthMode)
    }
    'video' {
        $appArgs += @('--path', $resolvedVideo)
    }
}
if ($DurationS -gt 0) {
    $appArgs += @('--duration-s', "$DurationS")
}
if ($ShadowMaskCheckpoint) {
    if (-not $ShadowMaskLog) {
        throw "-ShadowMaskCheckpoint needs -ShadowMaskLog"
    }
    $resolvedShadowLog = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($ShadowMaskLog)
    $resolvedShadowLogDir = Split-Path -Parent $resolvedShadowLog
    New-Item -ItemType Directory -Force -Path $resolvedShadowLogDir | Out-Null
    $appArgs += @(
        '--shadow-mask-checkpoint', (Resolve-Path $ShadowMaskCheckpoint).Path,
        '--shadow-mask-log', $resolvedShadowLog,
        '--shadow-mask-sha256', 'd821ffcc66557047b50625c4c7345634aacf72b55b0181d0d34d7e487d927606'
    )
}
if ($FaceDepthCorrection) {
    if (-not $FaceDepthRuleJson) {
        throw "-FaceDepthCorrection needs -FaceDepthRuleJson"
    }
    if (-not $FaceDepthLog) {
        throw "-FaceDepthCorrection needs -FaceDepthLog"
    }
    $resolvedFaceDepthLog = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($FaceDepthLog)
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $resolvedFaceDepthLog) | Out-Null
    $appArgs += @(
        '--face-depth-correction',
        '--face-depth-rule-json', (Resolve-Path $FaceDepthRuleJson).Path,
        '--face-depth-log', $resolvedFaceDepthLog
    )
}

$url = "http://127.0.0.1:$Port/"
Write-Step "Starting the pipeline ($Source) on $url"

# Pass every argument separately. Building a quoted command string breaks when a
# path contains spaces, because the nested quotes are consumed by the shell.
$quotedArgs = ($appArgs | ForEach-Object { "'" + ($_ -replace "'", "''") + "'" }) -join ' '
$innerCommand = @(
    '$ErrorActionPreference = ''Continue''',
    "& '$python' $quotedArgs 2>&1 | Tee-Object -FilePath '$outLog'",
    'Write-Host ""',
    'Write-Host "Pipeline stopped. Press Enter to close this window." -ForegroundColor Yellow',
    $(if ($DurationS -le 0) { 'Read-Host | Out-Null' })
) -join [Environment]::NewLine
$encodedCommand = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($innerCommand))
$windowArgs = @('-NoProfile', '-EncodedCommand', $encodedCommand)

$startOptions = @{
    FilePath = 'powershell.exe'
    ArgumentList = $windowArgs
    PassThru = $true
}
if ($NoBrowser -and $DurationS -gt 0) {
    $startOptions.WindowStyle = 'Hidden'
}
$appProcess = Start-Process @startOptions
Write-Step "Pipeline window started (PID $($appProcess.Id)). Log: $outLog"

# Wait for the app to answer, then for one published splat state. The first
# refresh after model load takes several seconds because of CUDA warm-up.
Write-Step 'Waiting for the server to answer'
$deadline = (Get-Date).AddSeconds($TimeoutSec)
$serverUp = $false
while ((Get-Date) -lt $deadline) {
    try {
        $response = Invoke-WebRequest -Uri "$url/api/health" -UseBasicParsing -TimeoutSec 2
        if ($response.StatusCode -eq 200) { $serverUp = $true; break }
    }
    catch {
        Start-Sleep -Milliseconds 500
    }
}
if (-not $serverUp) {
    throw "The server did not answer within $TimeoutSec seconds. Check the pipeline window and $outLog."
}
Write-Step 'Server is up'

if ($Source -ne 'video') {
    Write-Step 'Waiting for the first splat state (model warm-up can take about 5 seconds)'
    $stateSeen = $false
    while ((Get-Date) -lt $deadline) {
        try {
            $diag = (Invoke-WebRequest -Uri "$url/api/diagnostics" -UseBasicParsing -TimeoutSec 2).Content | ConvertFrom-Json
            if ($null -ne $diag.splat) {
                $stateSeen = $true
                break
            }
        }
        catch {
            # Keep waiting; the worker may still be loading models.
        }
        Start-Sleep -Milliseconds 700
    }
    if ($stateSeen) {
        Write-Step "First splat state published: frame $($diag.splat.frame_id), $($diag.splat.count) splats"
    }
    else {
        Write-Warning 'No splat state was published within the timeout. Is a person in view? The viewer opens anyway.'
    }
}

# ---------------------------------------------------------------- the viewer

if (-not $NoBrowser) {
    Write-Step "Opening the viewer at $url"
    Start-Process $url
}
else {
    Write-Host "[start] Viewer not opened (-NoBrowser). Open $url yourself." -ForegroundColor Yellow
}

Write-Host ''
Write-Host 'Live splat pipeline is running.' -ForegroundColor Green
Write-Host "  Viewer : $url"
Write-Host "  App PID: $($appProcess.Id)"
Write-Host "  Log    : $outLog"
if ($DurationS -gt 0) {
    Write-Host "  Stops  : after $DurationS seconds"
}
else {
    Write-Host '  Stop   : press Ctrl+C in the pipeline window'
}
Write-Host ''
Write-Host 'Tips:'
Write-Host '  - Leave the frame and press "Calibrate background" only when the scene is empty.'
Write-Host '  - The first refresh after launch takes several seconds (CUDA warm-up).'
Write-Host '  - Use the Mask tab to check what the person mask selects.'
