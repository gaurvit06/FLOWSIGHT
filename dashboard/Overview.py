"""FlowSight dashboard: Overview page.

    streamlit run dashboard/Overview.py
Needs the API running. Other pages live in dashboard/pages/.
"""

from __future__ import annotations

import time

import pandas as pd
import plotly.express as px
import streamlit as st
from common import (
    API,
    DEMO,
    LABEL_COLORS,
    LIVE_DEFAULT,
    PLOTLY_LAYOUT,
    SEV_COLORS,
    get,
    httpx,
    kpi,
    logo,
    setup,
    story_card,
)

setup("Overview")
logo()

with st.sidebar:
    auto = st.toggle("live mode (refresh 3 s)", value=LIVE_DEFAULT)
    if not DEMO and st.button("clear history", width="stretch"):
        httpx.delete(f"{API}/predictions")
        st.cache_data.clear()

s = get("/stats")
incidents = get("/incidents")
fb = get("/feedback/summary")
drift = get("/drift")

# ---- drift status line (U7)
if drift["status"] == "drift":
    st.markdown('<div class="fs-banner" style="border-color:#ff3b5c;color:#ff3b5c">'
                f'[ALERT] drift-monitor: {len(drift["drifted"])} feature(s) shifted from the '
                f'training data ({", ".join(drift["drifted"][:4])}). Predictions may be less '
                'reliable. Consider retraining.</div>', unsafe_allow_html=True)
elif drift["status"] == "warning":
    st.markdown('<div class="fs-banner" style="border-color:#ffd000;color:#ffd000">'
                f'[WARN] drift-monitor: features starting to shift '
                f'({", ".join(drift["warning"][:4])}).</div>', unsafe_allow_html=True)
elif drift["status"] == "stable":
    st.markdown('<div class="fs-banner" style="border-color:#3ddc97;color:#3ddc97">'
                f'[ OK ] drift-monitor: traffic matches the training distribution '
                f'(last {drift["n_flows"]} flows).</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="fs-banner" style="border-color:#5f7d95;color:#5f7d95">'
                f'[WAIT] drift-monitor: collecting flows ({drift["n_flows"]}/'
                f'{drift.get("min_flows", 100)}).</div>', unsafe_allow_html=True)

# ---- KPIs
active = [i for i in incidents if i["status"] == "active"]
c1, c2, c3, c4, c5 = st.columns(5)
kpi(c1, "Flows", f"{s['total']:,}")
kpi(c2, "Alerts", f"{s['attacks']:,}",
    f"{(s['attacks'] / s['total'] * 100) if s['total'] else 0:.1f}% of traffic", "#f97316")
kpi(c3, "Incidents", len(incidents), f"{len(active)} active", "#22d3ee")
kpi(c4, "Critical", s["by_severity"].get("Critical", 0), "", SEV_COLORS["Critical"])
kpi(c5, "Precision",
    f"{fb['precision']:.0%}" if fb["precision"] is not None else "–",
    f"{fb['reviewed']} reviewed · {fb['unreviewed']} pending", "#22c55e")

st.write("")
if s["total"] == 0:
    st.info("No traffic yet. Run `python -m nids.capture.replay --n 500` in another terminal, "
            "or upload a file on the **Analyse File** page.")
else:
    left, right = st.columns([2.2, 1])
    tl = pd.DataFrame(s["timeline"])
    fig = px.bar(tl, x="minute", y="n", color="label", color_discrete_map=LABEL_COLORS,
                 title="Traffic per minute by verdict")
    fig.update_layout(**PLOTLY_LAYOUT, height=340, legend_title=None, xaxis_title=None,
                      yaxis_title="flows")
    left.plotly_chart(fig, width="stretch")

    sev = pd.DataFrame([{"severity": k or "n/a", "n": v} for k, v in s["by_severity"].items()])
    if not sev.empty:
        fig = px.pie(sev, names="severity", values="n", hole=0.6, color="severity",
                     color_discrete_map=SEV_COLORS, title="Alerts by severity")
        fig.update_layout(**PLOTLY_LAYOUT, height=340, legend_title=None)
        right.plotly_chart(fig, width="stretch")

    st.subheader("Latest high-priority alerts")
    alerts = get("/predictions", limit=200, attacks_only=True)
    top = [a for a in alerts if a.get("severity") in ("Critical", "High")][:5] or alerts[:5]
    for a in top:
        story_card(a)

if auto:
    time.sleep(3)
    st.cache_data.clear()
    st.rerun()
