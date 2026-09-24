param(
    [string]$Compiler = "C:\msys64\mingw64\bin\g++.exe",
    [string]$BuildDirectory = "build\splat_constructor"
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path -LiteralPath $Compiler)) {
    throw "C++ compiler not found at $Compiler. Pass -Compiler with a C++20 compiler path."
}

$resolvedBuild = [System.IO.Path]::GetFullPath((Join-Path $PWD $BuildDirectory))
$workspace = [System.IO.Path]::GetFullPath($PWD)
if (-not $resolvedBuild.StartsWith($workspace, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "BuildDirectory must remain inside the workspace."
}
New-Item -ItemType Directory -Force -Path $resolvedBuild | Out-Null

$include = "modules\splat_constructor\include"
$sources = @(
    "modules\splat_constructor\src\contracts.cpp",
    "modules\splat_constructor\src\constructor.cpp",
    "modules\splat_constructor\src\image_io.cpp",
    "modules\splat_constructor\src\ply_writer.cpp"
)
$warnings = @("-Wall", "-Wextra", "-Wpedantic", "-Werror")

& $Compiler -std=c++20 -O2 @warnings "-I$include" @sources `
    "modules\splat_constructor\cli\main.cpp" `
    -o (Join-Path $resolvedBuild "splat_constructor_cli.exe")
if ($LASTEXITCODE -ne 0) { throw "Failed to build Splat Constructor CLI." }

& $Compiler -std=c++20 -O2 @warnings "-I$include" @sources `
    "modules\splat_constructor\tests\test_constructor.cpp" `
    -o (Join-Path $resolvedBuild "splat_constructor_tests.exe")
if ($LASTEXITCODE -ne 0) { throw "Failed to build Splat Constructor tests." }

Write-Host "Built Splat Constructor in $resolvedBuild"
