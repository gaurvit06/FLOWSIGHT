# Stops everything scripts\run_demo.ps1 started (API, replay, dashboard, tunnel).
#   powershell -ExecutionPolicy Bypass -File scripts\stop_demo.ps1

Get-CimInstance Win32_Process -Filter "Name like 'python%' or Name = 'ssh.exe'" |
    Where-Object { $_.CommandLine -match 'uvicorn|streamlit|nids\.capture\.replay|serveo\.net|localhost\.run|pinggy\.io' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force; "stopped $($_.ProcessId)" }
