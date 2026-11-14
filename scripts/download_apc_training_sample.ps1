param(
    [string]$ArchivePath = "data/processed/apc_training_sample.zip",
    [string]$Destination = "data/processed/apc_training_sample"
)

$ErrorActionPreference = "Stop"
$expectedSha256 = "AF9B5A801C51EAEB6228804B4388A68F00AFC17774DB5648EB95636495B84335"
$sourceUrl = "https://3dvision.princeton.edu/projects/2016/apc/downloads/training-sample.zip"

$archiveParent = Split-Path -Parent $ArchivePath
if ($archiveParent) {
    New-Item -ItemType Directory -Force -Path $archiveParent | Out-Null
}

if (-not (Test-Path -LiteralPath $ArchivePath)) {
    Invoke-WebRequest -Uri $sourceUrl -OutFile $ArchivePath
}

$actualSha256 = (Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256).Hash
if ($actualSha256 -ne $expectedSha256) {
    throw "Dataset archive checksum mismatch. Expected $expectedSha256, got $actualSha256."
}

New-Item -ItemType Directory -Force -Path $Destination | Out-Null
Expand-Archive -LiteralPath $ArchivePath -DestinationPath $Destination -Force
Write-Host "Dataset ready at $Destination/training-sample"
