# Capture a screenshot of the running GUI with headless Firefox and Selenium.
#
# Firefox runs with an isolated temporary profile, so the capture never uses or
# stops the user's own browser windows. The Python helper waits until the page
# shows a live frame, waits the requested time, optionally clicks a tab, and
# saves a PNG. It replaces the earlier Edge/DevTools capture, which terminated
# every running Microsoft Edge process.
#
#   ./tools/capture-screenshot.ps1 -Url http://127.0.0.1:5180/ `
#       -Out screenshots/gui-live-main.png -WaitMs 10000
#
# Requires Firefox and the `selenium` Python package in the repository venv.
param(
    [string]$Url = 'http://127.0.0.1:5180/',
    [string]$Out = 'screenshots/gui-live-main.png',
    [int]$WaitMs = 6000,
    [string]$ClickSelector = ''
)

$root = Resolve-Path (Join-Path $PSScriptRoot '..\..\..')
$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    throw "Python venv was not found at $python"
}
$helper = Join-Path $PSScriptRoot 'capture-firefox.py'
$arguments = @($helper, '--url', $Url, '--out', $Out, '--wait-ms', $WaitMs)
if ($ClickSelector) {
    $arguments += @('--click-selector', $ClickSelector)
}
& $python @arguments
exit $LASTEXITCODE
