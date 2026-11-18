param(
    [switch]$WithRaft,
    [switch]$SkipNative,
    [ValidateSet('auto', 'cpu', 'cu118', 'cu126', 'cu128')]
    [string]$TorchWheel = 'auto'
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repoRoot '.venv\Scripts\python.exe'

function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Executable $($Arguments -join ' ')"
    }
}

function Get-TorchWheel {
    param([string]$Requested)
    if ($Requested -ne 'auto') { return $Requested }

    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($null -eq $nvidiaSmi) { return 'cpu' }
    $report = & $nvidiaSmi.Source 2>$null | Out-String
    if ($LASTEXITCODE -ne 0 -or $report -notmatch 'CUDA (?:UMD )?Version:\s*(\d+)\.(\d+)') {
        return 'cpu'
    }
    $supportedVersion = [version]"$($Matches[1]).$($Matches[2])"
    if ($supportedVersion -ge [version]'12.8') { return 'cu128' }
    if ($supportedVersion -ge [version]'12.6') { return 'cu126' }
    if ($supportedVersion -ge [version]'11.8') { return 'cu118' }
    return 'cpu'
}

Push-Location $repoRoot
try {
    if (-not (Test-Path -LiteralPath $venvPython)) {
        $launcher = Get-Command py -ErrorAction SilentlyContinue
        $venvArguments = @('-3', '-m', 'venv', '.venv')
        if ($null -eq $launcher) {
            $launcher = Get-Command python -ErrorAction SilentlyContinue
            $venvArguments = @('-m', 'venv', '.venv')
        }
        if ($null -eq $launcher) { throw 'Install Python 3.11 or newer, then rerun this script.' }
        Write-Host 'Creating .venv with Python 3...'
        Invoke-Checked -Executable $launcher.Source -Arguments $venvArguments
    }

    $checkScript = Join-Path $PSScriptRoot 'check_device_setup.py'
    if (-not (Test-Path -LiteralPath $checkScript)) {
        throw "Setup check script not found: $checkScript"
    }
    Invoke-Checked -Executable $venvPython -Arguments @($checkScript, 'python')
    Write-Host 'Installing local pipeline modules...'
    Invoke-Checked -Executable $venvPython -Arguments @(
        '-m', 'pip', 'install',
        '-e', '.\modules\stream_handler',
        '-e', '.\modules\inference\optical_flow',
        '-e', '.\modules\reranker'
    )

    if ($WithRaft) {
        $selectedWheel = Get-TorchWheel $TorchWheel
        Write-Host "Installing PyTorch and Torchvision from the $selectedWheel wheel index..."
        Invoke-Checked -Executable $venvPython -Arguments @(
            '-m', 'pip', 'install', 'torch', 'torchvision',
            '--index-url', "https://download.pytorch.org/whl/$selectedWheel"
        )
        Invoke-Checked -Executable $venvPython -Arguments @($checkScript, 'torch')
    }

    Invoke-Checked -Executable $venvPython -Arguments @($checkScript, 'imports')

    if (-not $SkipNative) {
        Write-Host 'Building and testing the native Splat Constructor...'
        $buildScript = Join-Path $repoRoot 'scripts\build_splat_constructor.ps1'
        if (-not (Test-Path -LiteralPath $buildScript)) {
            throw "Native build script not found: $buildScript"
        }
        & $buildScript
        if ($LASTEXITCODE -ne 0) {
            throw "Native Splat Constructor build failed with exit code $LASTEXITCODE."
        }
        $nativeTests = Join-Path $repoRoot 'build\splat_constructor\splat_constructor_tests.exe'
        if (-not (Test-Path -LiteralPath $nativeTests)) {
            throw "Native test executable was not produced: $nativeTests"
        }
        & $nativeTests
        if ($LASTEXITCODE -ne 0) {
            throw "Native Splat Constructor tests failed with exit code $LASTEXITCODE."
        }
        Write-Host 'Native Splat Constructor build and tests passed.'
    } else {
        Write-Host 'Skipped the native Splat Constructor build (-SkipNative).'
    }

    Write-Host 'Setup complete. DIS optical flow works on CPU; RAFT uses CUDA when the installed PyTorch build and driver support it.'
}
finally {
    Pop-Location
}
