"""Entrypoint for the FlowSight Hugging Face Space (also runnable locally for testing).

Starts, in one container:
  1. the FastAPI service on localhost (never exposed),
  2. the Streamlit console on $PORT (the only public port), in public-demo mode,
  3. a never-ending stream of real CIC-IDS2017 flows into the API, so the console is live.

If the API or the console dies, the script exits with an error so the platform restarts it.
History is trimmed periodically so a two-week run doesn't grow the database without bound.
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
API_PORT = int(os.environ.get("NIDS_API_PORT", "8000"))
UI_PORT = int(os.environ.get("PORT", "8501"))
API = f"http://127.0.0.1:{API_PORT}"
MAX_HISTORY = int(os.environ.get("FLOWSIGHT_MAX_HISTORY", "20000"))

ENV = {
    **os.environ,
    "PYTHONPATH": str(ROOT / "src"),
    "FLOWSIGHT_DEMO": "1",
    "NIDS_API": API,
    "NIDS_DB_PATH": str(ROOT / "data" / "nids.db"),
    "NIDS_MODELS_DIR": str(ROOT / "models"),
}


def log(msg: str) -> None:
    print(f"[flowsight] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def wait_for(url: str, seconds: int = 180) -> None:
    for _ in range(seconds * 2):
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except Exception:  # noqa: BLE001
            time.sleep(0.5)
    raise SystemExit(f"Timed out waiting for {url}")


def api_call(path: str, method: str = "GET"):
    req = urllib.request.Request(f"{API}{path}", method=method)
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read() or b"null")


def replay(n: int, delay: float, seed: int) -> None:
    subprocess.run([sys.executable, "-m", "nids.capture.replay", "--api", API, "--n", str(n),
                    "--delay", str(delay), "--attack-frac", "0.25", "--seed", str(seed)],
                   env=ENV, cwd=ROOT, stdout=subprocess.DEVNULL, check=False)


def main() -> None:
    # --backend-only: API + traffic stream, no console. Used when the hosting platform runs
    # the Streamlit console itself (Streamlit Community Cloud); dashboard/common.py starts us.
    backend_only = "--backend-only" in sys.argv
    (ROOT / "data").mkdir(exist_ok=True)
    log(f"starting API on {API}")
    api = subprocess.Popen([sys.executable, "-m", "uvicorn", "nids.api.main:app",
                            "--host", "127.0.0.1", "--port", str(API_PORT)], env=ENV, cwd=ROOT)
    wait_for(f"{API}/health")

    ui = None
    if not backend_only:
        ui = start_console()

    api_call("/predictions", "DELETE")
    log("seeding history with 300 real flows")
    replay(300, 0.0, seed=0)

    for cycle in itertools.count(1):
        if api.poll() is not None or (ui is not None and ui.poll() is not None):
            raise SystemExit("API or console stopped; exiting so the platform restarts us")
        if api_call("/stats")["total"] > MAX_HISTORY:
            log("trimming history")
            api_call("/predictions", "DELETE")
            replay(300, 0.0, seed=cycle)
        replay(200, 1.5, seed=cycle)            # ~1 flow every 1.5 s, forever


def start_console() -> subprocess.Popen:
    log(f"starting console on 0.0.0.0:{UI_PORT}")
    ui = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", "dashboard/Overview.py",
        "--server.port", str(UI_PORT), "--server.address", "0.0.0.0",
        "--server.headless", "true", "--server.maxUploadSize", "20",
        # the Space is shown inside an iframe on huggingface.co; XSRF cookies don't survive
        # third-party iframes, which breaks uploads. Demo mode never stores uploads.
        "--server.enableXsrfProtection", "false", "--server.enableCORS", "false",
    ], env=ENV, cwd=ROOT)
    wait_for(f"http://127.0.0.1:{UI_PORT}/_stcore/health")
    return ui


if __name__ == "__main__":
    main()
