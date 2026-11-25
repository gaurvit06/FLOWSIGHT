"""Shared look and API helpers for every dashboard page.

Visual identity: "FlowSight SOC console": a security-operations terminal look of its own
(monospace type, cyan/amber on near-black, shell-prompt page headers, log-style alerts,
scanlines, a fixed status bar). All styling lives in this file and .streamlit/config.toml.
"""

from __future__ import annotations

import html
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "src").is_dir() and str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))          # hosted bundle: nids isn't pip-installed

API_PORT = os.environ.get("NIDS_API_PORT", "8000")
API = os.environ.get("NIDS_API", f"http://127.0.0.1:{API_PORT}")
# Hosted bundles (scripts/build_space.py) contain start.py and a DEMO_MODE marker file
EMBEDDED = (ROOT / "start.py").exists()
DEMO = (os.environ.get("FLOWSIGHT_DEMO", "").lower() in {"1", "true", "yes"}
        or (ROOT / "DEMO_MODE").exists())
# Live mode (auto-refresh) starts on in demo mode; FLOWSIGHT_LIVE_DEFAULT=0 turns that off
# (used by scripts/check_dashboard.py, which needs each page run to finish).
LIVE_DEFAULT = DEMO and os.environ.get("FLOWSIGHT_LIVE_DEFAULT", "1") == "1"

# ---- palette
BG, PANEL, BORDER = "#05080f", "#0a1120", "#12324a"
CYAN, AMBER, TEXT, DIM = "#00e5ff", "#ffb000", "#cfe8f5", "#5f7d95"
SEV_COLORS = {"Critical": "#ff3b5c", "High": "#ff8a00", "Medium": "#ffd000", "Low": "#3ddc97"}
LABEL_COLORS = {
    "Benign": "#2f4a63", "DDoS": "#ff3b5c", "DoS": "#ff8a00", "PortScan": "#b18cff",
    "BruteForce": "#ffd000", "Unknown-Anomaly": CYAN,
}
MONO = "'JetBrains Mono', 'Cascadia Code', Consolas, monospace"
PLOTLY_LAYOUT = {
    "template": "plotly_dark", "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)",
    "margin": {"l": 10, "r": 10, "t": 44, "b": 10},
    "font": {"color": TEXT, "family": MONO, "size": 12},
    "title": {"font": {"color": CYAN, "size": 13}},
    "xaxis": {"gridcolor": "#0f2536", "zerolinecolor": "#12324a"},
    "yaxis": {"gridcolor": "#0f2536", "zerolinecolor": "#12324a"},
}

# Page title -> the shell command shown in its prompt header
COMMANDS = {
    "Overview": "flowsight status --live",
    "Incidents": "flowsight incidents --timeline",
    "Investigate · what-if": "flowsight whatif --alert",
    "Models": "flowsight models --compare",
    "Analyst feedback": "flowsight review --queue",
    "Analyse a file": "flowsight analyse ./flows.csv",
    "Method & limits": "man flowsight",
}

# FlowSight's ASCII logo, assembled from a small block font so the columns always line up
_FONT = {
    "F": [" ___ ", "| __|", "| _| ", "|_|  "],
    "L": [" _    ", "| |   ", "| |__ ", "|____|"],
    "O": ["  ___  ", " / _ \\ ", "| (_) |", " \\___/ "],
    "W": ["__      __", "\\ \\    / /", " \\ \\/\\/ / ", "  \\_/\\_/  "],
    "S": [" ___ ", "/ __|", "\\__ \\", "|___/"],
    "I": [" ___ ", "|_ _|", " | | ", "|___|"],
    "G": ["  ___ ", " / __|", "| (_ |", " \\___|"],
    "H": [" _  _ ", "| || |", "| __ |", "|_||_|"],
    "T": [" _____ ", "|_   _|", "  | |  ", "  |_|  "],
}
LOGO = "\n".join("".join(_FONT[c][row] for c in "FLOWSIGHT") for row in range(4))

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;800&display=swap');
html, body, [class*="st-"], .stMarkdown, .stDataFrame, button, input, textarea, select {{
  font-family: {MONO} !important; }}
.stApp {{ background: radial-gradient(1200px 600px at 80% -10%, #06202e 0%, {BG} 55%); }}
.stApp::after {{ content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 999;
  background: repeating-linear-gradient(0deg, rgba(0,229,255,.025) 0 1px, transparent 1px 3px); }}
.block-container {{ padding-top: 3.2rem; padding-bottom: 4rem; }}
section[data-testid="stSidebar"] {{ background: #040910; border-right: 1px solid {BORDER}; }}
h1, h2, h3 {{ font-family: {MONO} !important; letter-spacing: -.01em; }}
h2, h3 {{ color: {CYAN} !important; font-size: 1.05rem !important; text-transform: uppercase;
  letter-spacing: .08em !important; }}
h2::before, h3::before {{ content: "## "; color: {DIM}; }}

.fs-logo {{ color: {CYAN}; font-size: 1.05rem; line-height: 1.15; white-space: nowrap; margin: 0;
  font-family: {MONO};
  text-shadow: 0 0 12px rgba(0,229,255,.35); overflow-x: auto; }}
.fs-tagline {{ color: {AMBER}; font-size: .78rem; letter-spacing: .12em; margin: 6px 0 18px; }}
.fs-prompt {{ color: {DIM}; font-size: .85rem; margin-bottom: 2px; }}
.fs-prompt .u {{ color: #3ddc97; }} .fs-prompt .p {{ color: {CYAN}; }}
.fs-prompt .c {{ color: {TEXT}; }}
.fs-cursor {{ display: inline-block; width: .6em; background: {CYAN}; margin-left: 2px;
  animation: fs-blink 1.1s steps(1) infinite; }}
@keyframes fs-blink {{ 50% {{ opacity: 0; }} }}
.fs-h1 {{ font-size: 1.9rem; font-weight: 800; color: {TEXT}; margin: 0 0 14px;
  border-bottom: 1px dashed {BORDER}; padding-bottom: 10px; }}

.fs-brand {{ color: {CYAN}; font-weight: 800; letter-spacing: .14em; font-size: 1.05rem; }}
.fs-brand .dot {{ color: {AMBER}; }}
.fs-side {{ color: {DIM}; font-size: .74rem; line-height: 1.6; }}
.fs-side b {{ color: {TEXT}; font-weight: 600; }}
.fs-on {{ color: #3ddc97; }} .fs-off {{ color: #ff3b5c; }}

.fs-kpi {{ background: linear-gradient(180deg, {PANEL}, #070d18); border: 1px solid {BORDER};
  border-top: 2px solid var(--k, {CYAN}); padding: 12px 14px; height: 100%; }}
.fs-kpi .l {{ color: {DIM}; font-size: .68rem; text-transform: uppercase; letter-spacing: .05em;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.fs-kpi .l::before {{ content: ":: "; }}
.fs-kpi .v {{ font-size: 1.75rem; font-weight: 800; margin-top: 4px; }}
.fs-kpi .s {{ color: {DIM}; font-size: .72rem; margin-top: 2px; }}

.fs-pill {{ display: inline-block; padding: 0 6px; font-size: .72rem; font-weight: 800;
  color: {BG}; margin-right: 6px; }}
.fs-tag {{ display: inline-block; padding: 0 6px; font-size: .72rem; font-weight: 600;
  border: 1px solid; margin-right: 6px; text-transform: uppercase; }}

.fs-log {{ background: {PANEL}; border: 1px solid {BORDER}; border-left: 3px solid var(--c);
  padding: 10px 12px; margin-bottom: 8px; }}
.fs-log .hd {{ font-size: .76rem; color: {DIM}; }}
.fs-log .hd .sev {{ color: var(--c); font-weight: 800; }}
.fs-log .hd .lbl {{ color: {TEXT}; font-weight: 600; }}
.fs-log .msg {{ color: {TEXT}; font-size: .86rem; margin-top: 6px; line-height: 1.5; }}
.fs-log .msg::before {{ content: "> "; color: var(--c); }}

.fs-banner {{ padding: 8px 12px; margin: 4px 0 14px; font-size: .84rem; border: 1px solid;
  background: {PANEL}; }}
.fs-statusbar {{ position: fixed; left: 0; right: 0; bottom: 0; z-index: 1000001;
  background: #030609; border-top: 1px solid {BORDER}; color: {DIM}; font-size: .72rem;
  padding: 5px 16px; display: flex; gap: 18px; white-space: nowrap; overflow: hidden; }}
.fs-statusbar b {{ color: {CYAN}; font-weight: 600; }}
.fs-statusbar .amber {{ color: {AMBER}; }}
div[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; }}
.stButton > button {{ border-radius: 0; border: 1px solid {BORDER}; background: {PANEL}; }}
.stButton > button:hover {{ border-color: {CYAN}; color: {CYAN}; }}
</style>
"""


def _api_up() -> bool:
    try:
        return httpx.get(f"{API}/health", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


@st.cache_resource(show_spinner="Starting the FlowSight engine (first visit only, ~30 s)...")
def _start_backend():
    """Hosted single-process platforms (e.g. Streamlit Community Cloud) run only this app, so
    the bundle starts its own API + live traffic stream once per server process."""
    proc = subprocess.Popen([sys.executable, str(ROOT / "start.py"), "--backend-only"],
                            cwd=ROOT, env={**os.environ, "NIDS_API_PORT": API_PORT})
    for _ in range(360):
        if _api_up():
            return proc
        if proc.poll() is not None:
            break
        time.sleep(0.5)
    raise RuntimeError("FlowSight engine failed to start")


def setup(title: str) -> dict:
    st.set_page_config(page_title=f"FlowSight · {title}", page_icon="🛰️", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    # Hosted bundle: start the engine on first visit; if it died since, start a fresh one
    if EMBEDDED and not _api_up() and _start_backend().poll() is not None:
        _start_backend.clear()
        _start_backend()
    with st.sidebar:
        st.markdown('<div class="fs-brand">FLOWSIGHT<span class="dot">_</span></div>'
                    '<div class="fs-side">explainable network intrusion detection</div>',
                    unsafe_allow_html=True)
    try:
        health = get("/health")
    except Exception as e:  # noqa: BLE001
        st.error(f"Can't reach the API at {API}. Start it with:\n\n"
                 f"`uvicorn nids.api.main:app --port 8000`\n\n{e}")
        st.stop()
    det = health["anomaly_detector"]
    with st.sidebar:
        st.markdown(
            f'<div class="fs-side">api <span class="fs-on">● online</span><br>'
            f'model <b>{html.escape(health["model"])}</b><br>'
            f'unknown-attack rule <span class="{"fs-on" if det else "fs-off"}">'
            f'{"● armed" if det else "○ off"}</span>'
            f'{"<br>mode <b>public demo</b>" if DEMO else ""}</div>', unsafe_allow_html=True)
        if st.button("↻ refresh", width="stretch"):
            st.cache_data.clear()
    cmd = COMMANDS.get(title, f"flowsight {title.lower()}")
    st.markdown(f'<div class="fs-prompt"><span class="u">analyst@flowsight</span>:'
                f'<span class="p">~</span>$ <span class="c">{html.escape(cmd)}</span>'
                f'<span class="fs-cursor">&nbsp;</span></div>'
                f'<div class="fs-h1">{html.escape(title)}</div>', unsafe_allow_html=True)
    status_bar(health)
    return health


def logo() -> None:
    """The big ASCII logo (Overview page only)."""
    # Markdown collapses runs of spaces, so encode them explicitly to keep the letters aligned
    art = html.escape(LOGO).replace(" ", "&nbsp;").replace("\n", "<br>")
    st.markdown(f'<div class="fs-logo">{art}</div>'
                '<div class="fs-tagline">:: soc console // flow-level intrusion detection '
                '// every alert explained ::</div>', unsafe_allow_html=True)


def status_bar(health: dict) -> None:
    now = datetime.now(timezone.utc).strftime("%H:%M UTC")
    st.markdown(
        f'<div class="fs-statusbar"><span><b>FLOWSIGHT</b> // soc-console</span>'
        f'<span>model <b>{html.escape(health["model"])}</b></span>'
        f'<span>dataset <b>CIC-IDS2017</b></span>'
        f'<span>unknown-attack rule <b>{"armed" if health["anomaly_detector"] else "off"}</b>'
        f'</span>{"<span class=amber>PUBLIC DEMO</span>" if DEMO else ""}'
        f'<span style="margin-left:auto">{now}</span></div>', unsafe_allow_html=True)


@st.cache_data(ttl=3, show_spinner=False)
def get(path: str, **params):
    r = httpx.get(f"{API}{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def post(path: str, json=None, params=None):
    r = httpx.post(f"{API}{path}", json=json, params=params, timeout=300)
    r.raise_for_status()
    return r.json()


def kpi(col, label: str, value, sub: str = "", color: str = TEXT) -> None:
    col.markdown(f'<div class="fs-kpi" style="--k:{color}"><div class="l">{label}</div>'
                 f'<div class="v" style="color:{color}">{value}</div>'
                 f'<div class="s">{sub}</div></div>', unsafe_allow_html=True)


def sev_pill(sev: str | None) -> str:
    if not sev:
        return ""
    return f'<span class="fs-pill" style="background:{SEV_COLORS.get(sev, DIM)}">{sev.upper()}</span>'


def label_tag(label: str) -> str:
    c = LABEL_COLORS.get(label, "#94a3b8")
    return f'<span class="fs-tag" style="color:{c};border-color:{c}">{html.escape(label)}</span>'


def story_card(a: dict) -> None:
    """An alert rendered as a log entry."""
    sev = a.get("severity")
    color = SEV_COLORS.get(sev, DIM)
    verdict = {"confirmed": " · ✓ confirmed", "false_positive": " · ✗ false positive"}.get(
        a.get("verdict"), "")
    ts = a["ts"][11:19] + "Z"
    st.markdown(
        f'<div class="fs-log" style="--c:{color}"><div class="hd">'
        f'<span class="sev">[{(sev or "info").upper()}]</span> {ts} '
        f'<span class="lbl">{html.escape(a["label"])}</span> · alert#{a["id"]} · '
        f'incident#{a.get("incident_id") or "-"}{verdict}</div>'
        f'<div class="msg">{html.escape(a.get("story") or "")}</div></div>',
        unsafe_allow_html=True)


def shap_chart(explanation: list[dict], title: str = "why: feature contributions (SHAP)"):
    if not explanation:
        st.caption("No feature-level explanation for this prediction.")
        return
    ex = pd.DataFrame(explanation).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=ex["contribution"], y=ex["feature"], orientation="h",
        text=ex["value"].map(lambda v: f"value {v:,.2f}"), textposition="auto",
        marker_color=[SEV_COLORS["Critical"] if c > 0 else CYAN for c in ex["contribution"]]))
    fig.update_layout(**{**PLOTLY_LAYOUT, "title": {"text": title, **PLOTLY_LAYOUT["title"]}},
                      height=300, xaxis_title="contribution (+ pushes towards the verdict)")
    st.plotly_chart(fig, width="stretch")


def feedback_buttons(a: dict, key: str = "") -> None:
    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button("✓ confirm", key=f"ok{key}{a['id']}"):
        post(f"/feedback/{a['id']}", json={"verdict": "confirmed"})
        st.cache_data.clear()
        st.rerun()
    if c2.button("✗ false positive", key=f"fp{key}{a['id']}"):
        post(f"/feedback/{a['id']}", json={"verdict": "false_positive"})
        st.cache_data.clear()
        st.rerun()
