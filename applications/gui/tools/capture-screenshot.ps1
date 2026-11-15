# Capture a screenshot of the running GUI with headless Edge over the DevTools
# Protocol (CDP). This is the method used by the gsplat spike report:
# start Edge headless with a remote debugging port, navigate to the page with
# Page.navigate, wait real time, then call Page.captureScreenshot.
#
# Example:
#   ./tools/capture-screenshot.ps1 -Url http://127.0.0.1:5180/ `
#       -Out screenshots/gui-live.png
#
# Optional extras used for the per-tab verification shots:
#   -ClickSelector '.processor-tab[data-view="motion"]'
#   -WaitForExpression "document.querySelector('.processor-placeholder').style.display === 'none'"
param(
    [string]$Url = 'http://127.0.0.1:5180/',
    [string]$Out = 'screenshots/gui-live.png',
    [int]$WaitMs = 10000,
    [int]$Width = 1400,
    [int]$Height = 800,
    [string]$ClickSelector = '',
    [string]$WaitForExpression = '',
    [int]$WaitForTimeoutMs = 60000,
    [int]$PostDelayMs = 600,
    [int]$CommandTimeoutMs = 120000
)

$ErrorActionPreference = 'Stop'

$edge = 'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
if (-not (Test-Path $edge)) {
    throw "Edge was not found at $edge"
}

$userData = Join-Path $env:TEMP 'edge-gui'
if (Test-Path $userData) {
    Remove-Item -Recurse -Force $userData -ErrorAction SilentlyContinue
}

# Stop an earlier capture run before starting a new one.
Get-Process msedge -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -eq $edge } |
    Stop-Process -Force -ErrorAction SilentlyContinue

$argLine = @(
    '--headless=new'
    '--remote-debugging-port=9334'
    '--disable-gpu-sandbox'
    '--use-gl=angle'
    '--enable-unsafe-swiftshader'
    '--no-first-run'
    '--no-default-browser-check'
    "--window-size=$Width,$Height"
    "--user-data-dir=`"$userData`""
    'about:blank'
) -join ' '

Write-Output "STEP starting Edge"
$edgeProcess = Start-Process -FilePath $edge -ArgumentList $argLine -PassThru

function Wait-DevTools {
    for ($attempt = 0; $attempt -lt 150; $attempt++) {
        try {
            $targets = Invoke-RestMethod 'http://127.0.0.1:9334/json/list' -TimeoutSec 2
            $page = $targets | Where-Object { $_.type -eq 'page' } | Select-Object -First 1
            if ($page) {
                return $page
            }
        } catch {
            # The debugging endpoint is not up yet.
        }
        Start-Sleep -Milliseconds 200
    }
    throw 'The Edge DevTools endpoint did not become ready'
}

$page = Wait-DevTools
Write-Output "STEP devtools ready"

$ws = [System.Net.WebSockets.ClientWebSocket]::new()
$ws.ConnectAsync([Uri]$page.webSocketDebuggerUrl, [System.Threading.CancellationToken]::None).Wait()

$script:cdpId = 0
$receiveBuffer = New-Object byte[] 4194304

function Receive-CdpMessage {
    $stream = New-Object System.IO.MemoryStream
    do {
        $segment = [System.ArraySegment[byte]]::new($receiveBuffer)
        $task = $script:ws.ReceiveAsync($segment, [System.Threading.CancellationToken]::None)
        if (-not $task.Wait($script:CommandTimeoutMs)) {
            throw 'Timed out waiting for a DevTools message'
        }
        $received = $task.Result
        if ($received.Count -gt 0) {
            $stream.Write($receiveBuffer, 0, $received.Count)
        }
    } while (-not $received.EndOfMessage)
    $text = [System.Text.Encoding]::UTF8.GetString($stream.ToArray())
    $stream.Dispose()
    return ($text | ConvertFrom-Json)
}

function Send-Cdp([string]$method, $params, [bool]$expectResponse = $true) {
    $script:cdpId++
    $id = $script:cdpId
    $message = @{ id = $id; method = $method; params = $params } | ConvertTo-Json -Compress -Depth 20
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($message)
    $segment = [System.ArraySegment[byte]]::new($bytes)
    $script:ws.SendAsync(
        $segment,
        [System.Net.WebSockets.WebSocketMessageType]::Text,
        $true,
        [System.Threading.CancellationToken]::None
    ).Wait()
    if (-not $expectResponse) {
        return $null
    }
    while ($true) {
        $response = Receive-CdpMessage
        if ($response.id -eq $id) {
            if ($null -ne $response.error) {
                throw "CDP $method failed: $($response.error.message)"
            }
            return $response
        }
    }
}

function Evaluate-Page([string]$expression) {
    $result = Send-Cdp 'Runtime.evaluate' @{
        expression = $expression
        returnByValue = $true
        awaitPromise = $false
    }
    return $result.result.result.value
}

try {
    Send-Cdp 'Page.enable' @{} | Out-Null
    Send-Cdp 'Runtime.enable' @{} | Out-Null
    Send-Cdp 'Emulation.setDeviceMetricsOverride' @{
        width = $Width
        height = $Height
        deviceScaleFactor = 1
        mobile = $false
    } | Out-Null

    # Record page errors in the page itself, so one evaluate call can report them.
    Send-Cdp 'Page.addScriptToEvaluateOnNewDocument' @{
        source = @'
window.__captureErrors = [];
window.addEventListener('error', function (event) {
  window.__captureErrors.push('error: ' + event.message);
});
window.addEventListener('unhandledrejection', function (event) {
  var reason = event.reason;
  var stack = reason && reason.stack ? ' | ' + String(reason.stack).split('\n').slice(0, 5).join(' ') : '';
  window.__captureErrors.push('rejection: ' + String(reason) + stack);
});
'@
    } | Out-Null

    Write-Output "STEP navigating to $Url"
    Send-Cdp 'Page.navigate' @{ url = $Url } | Out-Null
    Start-Sleep -Milliseconds $WaitMs
    Write-Output 'STEP settle wait done'

    if ($ClickSelector -ne '') {
        $selectorJson = $ClickSelector | ConvertTo-Json -Compress
        $clicked = Evaluate-Page "(() => { const el = document.querySelector($selectorJson); if (!el) { return false } el.click(); return true })()"
        Write-Output "CLICK $ClickSelector -> $clicked"
        Start-Sleep -Milliseconds 400
    }

    if ($WaitForExpression -ne '') {
        $deadline = [DateTime]::UtcNow.AddMilliseconds($WaitForTimeoutMs)
        $satisfied = $false
        while ([DateTime]::UtcNow -lt $deadline) {
            if (Evaluate-Page "Boolean($WaitForExpression)") {
                $satisfied = $true
                break
            }
            Start-Sleep -Milliseconds 300
        }
        Write-Output "WAIT_EXPRESSION satisfied=$satisfied"
    }

    if ($PostDelayMs -gt 0) {
        Start-Sleep -Milliseconds $PostDelayMs
    }

    # Capture first: the live panels update every 300 ms, and for short-lived
    # states (the reuse motion field) the screenshot must win the race.
    Write-Output 'STEP capturing screenshot'
    $screenshot = Send-Cdp 'Page.captureScreenshot' @{ format = 'png' }
    $png = [Convert]::FromBase64String($screenshot.result.data)
    $outPath = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $Out))
    $outDirectory = [System.IO.Path]::GetDirectoryName($outPath)
    if (-not (Test-Path $outDirectory)) {
        New-Item -ItemType Directory -Force -Path $outDirectory | Out-Null
    }
    [System.IO.File]::WriteAllBytes($outPath, $png)
    Write-Output "SCREENSHOT: $outPath ($($png.Length) bytes)"

    Write-Output 'STEP reading page state'
    $expression = @'
JSON.stringify({
  title: document.title,
  statusLabel: (document.querySelector('.status-label') || {}).textContent,
  splatStatus: (document.querySelector('.splat-status') || {}).textContent,
  splatPlaceholder: (function () {
    var el = document.querySelector('.splat-placeholder');
    return el ? getComputedStyle(el).display : null
  })(),
  processorPlaceholder: (function () {
    var el = document.querySelector('.processor-placeholder');
    return el ? { display: getComputedStyle(el).display, text: el.textContent.trim() } : null
  })(),
  bodyText: document.body.innerText.replace(/\s+/g, ' ').slice(0, 4000),
  errors: window.__captureErrors ? window.__captureErrors.slice(0, 20) : [],
  canvases: Array.from(document.querySelectorAll('canvas')).map(function (c) {
    return { className: c.className, width: c.width, height: c.height,
             display: getComputedStyle(c).display,
             rect: (function (r) { return [Math.round(r.width), Math.round(r.height)] })(c.getBoundingClientRect()) }
  })
})
'@
    Write-Output 'PAGE_STATE_START'
    Write-Output (Evaluate-Page $expression)
    Write-Output 'PAGE_STATE_END'
} finally {
    try {
        Send-Cdp 'Browser.close' @{} $false
    } catch {
    }
    $ws.Dispose()
    Start-Sleep -Milliseconds 400
    if (-not $edgeProcess.HasExited) {
        Stop-Process -Id $edgeProcess.Id -Force -ErrorAction SilentlyContinue
    }
    Get-Process msedge -ErrorAction SilentlyContinue |
        Where-Object { $_.Path -eq $edge } |
        Stop-Process -Force -ErrorAction SilentlyContinue
}
