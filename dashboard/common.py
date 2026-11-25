"""Shared look and API helpers for every dashboard page."""

from __future__ import annotations

import html
import os

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

API = os.environ.get("NIDS_API", "http://localhost:8000")

SEV_COLORS = {"Critical": "#ef4444", "High": "#f97316", "Medium": "#eab308", "Low": "#22c55e"}
LABEL_COLORS = {
    "Benign": "#475569", "DDoS": "#ef4444", "DoS": "#f97316", "PortScan": "#a855f7",
    "BruteForce": "#eab308", "Unknown-Anomaly": "#22d3ee",
}
PLOTLY_LAYOUT = {"template": "plotly_dark", "paper_bgcolor": "rgba(0,0,0,0)",
                     "plot_bgcolor": "rgba(0,0,0,0)", "margin": {"l": 10, "r": 10, "t": 40, "b": 10},
                     "font": {"color": "#e5e7eb"}}

CSS = """
<style>
.block-container {padding-top: 1.6rem;}
.fs-brand {font-size: 1.35rem; font-weight: 700; letter-spacing: .02em;}
.fs-brand span {color: #22d3ee;}
.fs-kpi {background: #131a2e; border: 1px solid #1f2a44; border-radius: 12px;
         padding: 14px 16px; height: 100%;}
.fs-kpi .l {color: #94a3b8; font-size: .78rem; text-transform: uppercase; letter-spacing: .06em;
           white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
.fs-kpi .v {font-size: 1.8rem; font-weight: 700; margin-top: 2px;}
.fs-kpi .s {color: #94a3b8; font-size: .8rem;}
.fs-pill {display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .75rem;
          font-weight: 700; color: #0b1020; margin-right: 6px;}
.fs-tag {display: inline-block; padding: 2px 8px; border-radius: 6px; font-size: .75rem;
         font-weight: 600; border: 1px solid; margin-right: 6px;}
.fs-card {background: #131a2e; border-left: 4px solid var(--c); border-radius: 10px;
          padding: 12px 14px; margin-bottom: 10px;}
.fs-card .meta {color: #94a3b8; font-size: .78rem; margin-top: 6px;}
.fs-banner {border-radius: 10px; padding: 10px 14px; margin: 6px 0 14px 0; font-size: .92rem;}
</style>
"""


def setup(title: str) -> dict:
    st.set_page_config(page_title=f"FlowSight · {title}", page_icon="🛰️", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown('<div class="fs-brand">Flow<span>Sight</span></div>', unsafe_allow_html=True)
        st.caption("Explainable network intrusion detection")
    try:
        health = get("/health")
    except Exception as e:  # noqa: BLE001
        st.error(f"Can't reach the API at {API}. Start it with:\n\n"
                 f"`uvicorn nids.api.main:app --port 8000`\n\n{e}")
        st.stop()
    with st.sidebar:
        st.caption(f"Model: **{health['model']}** · unknown-attack detector: "
                   f"**{'on' if health['anomaly_detector'] else 'off'}**")
        if st.button("↻ Refresh", width="stretch"):
            st.cache_data.clear()
    st.title(title)
    return health


@st.cache_data(ttl=3, show_spinner=False)
def get(path: str, **params):
    r = httpx.get(f"{API}{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def post(path: str, json=None, params=None):
    r = httpx.post(f"{API}{path}", json=json, params=params, timeout=300)
    r.raise_for_status()
    return r.json()


def kpi(col, label: str, value, sub: str = "", color: str = "#e5e7eb") -> None:
    col.markdown(f'<div class="fs-kpi"><div class="l">{label}</div>'
                 f'<div class="v" style="color:{color}">{value}</div>'
                 f'<div class="s">{sub}</div></div>', unsafe_allow_html=True)


def sev_pill(sev: str | None) -> str:
    if not sev:
        return ""
    return f'<span class="fs-pill" style="background:{SEV_COLORS.get(sev, "#64748b")}">{sev}</span>'


def label_tag(label: str) -> str:
    c = LABEL_COLORS.get(label, "#94a3b8")
    return f'<span class="fs-tag" style="color:{c};border-color:{c}">{html.escape(label)}</span>'


def story_card(a: dict) -> None:
    color = SEV_COLORS.get(a.get("severity"), "#475569")
    verdict = {"confirmed": " · ✅ confirmed", "false_positive": " · ❌ false positive"}.get(
        a.get("verdict"), "")
    st.markdown(
        f'<div class="fs-card" style="--c:{color}">{sev_pill(a.get("severity"))}'
        f'{label_tag(a["label"])}<div style="margin-top:8px">{html.escape(a.get("story") or "")}'
        f'</div><div class="meta">alert #{a["id"]} · {a["ts"][:19].replace("T", " ")} UTC · '
        f'incident #{a.get("incident_id") or "-"}{verdict}</div></div>',
        unsafe_allow_html=True)


def shap_chart(explanation: list[dict], title: str = "What pushed the model to this verdict"):
    if not explanation:
        st.caption("No feature-level explanation for this prediction.")
        return
    ex = pd.DataFrame(explanation).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=ex["contribution"], y=ex["feature"], orientation="h",
        text=ex["value"].map(lambda v: f"value {v:,.2f}"), textposition="auto",
        marker_color=["#ef4444" if c > 0 else "#38bdf8" for c in ex["contribution"]]))
    fig.update_layout(**PLOTLY_LAYOUT, title=title, height=300,
                      xaxis_title="contribution (+ pushes towards the verdict)")
    st.plotly_chart(fig, width="stretch")


def feedback_buttons(a: dict, key: str = "") -> None:
    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button("✅ Confirm", key=f"ok{key}{a['id']}"):
        post(f"/feedback/{a['id']}", json={"verdict": "confirmed"})
        st.cache_data.clear()
        st.rerun()
    if c2.button("❌ False positive", key=f"fp{key}{a['id']}"):
        post(f"/feedback/{a['id']}", json={"verdict": "false_positive"})
        st.cache_data.clear()
        st.rerun()
