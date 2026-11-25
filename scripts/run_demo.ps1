# FlowSight public demo launcher (Windows PowerShell)
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_demo.ps1
#
# Starts the API (private, localhost only), streams real CIC-IDS2017 flows through it, starts
# the dashboard in demo mode (no history clearing, no exports, small uploads) and opens a
# public HTTPS tunnel to the dashboard with serveo.net over SSH (no install needed).
# The free tier gives a random URL that changes each run. Everything keeps running in the
# background until you run scripts\stop_demo.ps1.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
$logs = Join-Path $root "data\logs"
New-Item -ItemType Directory -Force $logs | Out-Null

function Wait-Url($url) {
    for ($i = 0; $i -lt 120; $i++) {
        try { Invoke-WebRequest -UseBasicParsing $url -TimeoutSec 2 | Out-Null; return }
        catch { Start-Sleep -Milliseconds 500 }
    }
    throw "Timed out waiting for $url"
}

# Stop anything left from a previous run
Get-CimInstance Win32_Process -Filter "Name like 'python%'" |
    Where-Object { $_.CommandLine -match 'uvicorn|streamlit|nids\.capture\.replay' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force }

Write-Host "Starting API (localhost:8000, not exposed)..."
Start-Process $py -ArgumentList "-m", "uvicorn", "nids.api.main:app", "--host", "127.0.0.1",
    "--port", "8000" -WorkingDirectory $root -WindowStyle Hidden `
    -RedirectStandardOutput "$logs\api.log" -RedirectStandardError "$logs\api.err.log"
Wait-Url "http://127.0.0.1:8000/health"
Invoke-RestMethod -Method Delete "http://127.0.0.1:8000/predictions" | Out-Null

Write-Host "Streaming real CIC-IDS2017 flows into the API (about 1 per second)..."
Start-Process $py -ArgumentList "-m", "nids.capture.replay", "--n", "4000", "--delay", "1",
    "--attack-frac", "0.25" -WorkingDirectory $root -WindowStyle Hidden `
    -RedirectStandardOutput "$logs\replay.log" -RedirectStandardError "$logs\replay.err.log"

Write-Host "Starting dashboard in demo mode (localhost:8501)..."
$env:FLOWSIGHT_DEMO = "1"
Start-Process $py -ArgumentList "-m", "streamlit", "run", "dashboard/Overview.py",
    "--server.port", "8501", "--server.maxUploadSize", "20" -WorkingDirectory $root `
    -WindowStyle Hidden -RedirectStandardOutput "$logs\dashboard.log" `
    -RedirectStandardError "$logs\dashboard.err.log"
Remove-Item Env:\FLOWSIGHT_DEMO
Wait-Url "http://127.0.0.1:8501"

Write-Host "Opening public tunnel (serveo.net over SSH)..."
# serveo prints "Forwarding HTTP traffic from https://..." even without a terminal.
# (localhost.run's free tier was too slow to deliver Streamlit's JavaScript; Pinggy only
#  shows its URL in an interactive terminal.)
Remove-Item "$logs\tunnel.log", "$logs\public_url.txt" -ErrorAction SilentlyContinue
Start-Process ssh -ArgumentList "-T", "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ServerAliveInterval=30", "-o", "ExitOnForwardFailure=yes",
    "-R", "80:127.0.0.1:8501", "serveo.net" `
    -WindowStyle Hidden -RedirectStandardOutput "$logs\tunnel.log" `
    -RedirectStandardError "$logs\tunnel.err.log"
$url = $null
for ($i = 0; $i -lt 60 -and -not $url; $i++) {
    Start-Sleep 1
    $text = Get-Content "$logs\tunnel.log" -Raw -ErrorAction SilentlyContinue
    if ($text) {
        $m = [regex]::Match($text, 'https://[a-z0-9\-\.]+serveo[a-z\.]*')
        if ($m.Success) { $url = $m.Value }
    }
}
if (-not $url) { throw "Tunnel did not report a URL; see $logs\tunnel.log and tunnel.err.log" }
$url | Set-Content -Encoding ascii "$logs\public_url.txt"
Write-Host ""
Write-Host "PUBLIC URL: $url"
Write-Host "Anyone with this URL can see the dashboard while your computer runs the demo."
Write-Host "Stop everything with: powershell -ExecutionPolicy Bypass -File scripts\stop_demo.ps1"
