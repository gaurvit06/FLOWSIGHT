"""FlowSight dashboard: Overview page.

    streamlit run dashboard/app.py
Needs the API running. Other pages live in dashboard/pages/.
"""

from __future__ import annotations

import time

import pandas as pd
import plotly.express as px
import streamlit as st
from common import (
    API,
    LABEL_COLORS,
    PLOTLY_LAYOUT,
    SEV_COLORS,
    get,
    httpx,
    kpi,
    setup,
    story_card,
)

setup("Overview")

with st.sidebar:
    auto = st.toggle("Live mode (refresh every 3 s)", value=False)
    if st.button("Clear history", width="stretch"):
        httpx.delete(f"{API}/predictions")
        st.cache_data.clear()

s = get("/stats")
incidents = get("/incidents")
fb = get("/feedback/summary")
drift = get("/drift")

# ---- drift banner (U7)
if drift["status"] == "drift":
    st.markdown(f'<div class="fs-banner" style="background:#3b1d1d;border:1px solid #ef4444">'
                f'⚠️ <b>Traffic has drifted from the training data</b>: '
                f'{len(drift["drifted"])} feature(s) shifted ({", ".join(drift["drifted"][:4])}). '
                f'Predictions may be less reliable. Consider retraining.</div>',
                unsafe_allow_html=True)
elif drift["status"] == "warning":
    st.markdown('<div class="fs-banner" style="background:#3a2f12;border:1px solid #eab308">'
                f'Some features are starting to shift ({", ".join(drift["warning"][:4])}).</div>',
                unsafe_allow_html=True)
elif drift["status"] == "stable":
    st.markdown('<div class="fs-banner" style="background:#11291f;border:1px solid #22c55e">'
                f'✓ Traffic matches the training distribution (last {drift["n_flows"]} flows).'
                '</div>', unsafe_allow_html=True)

# ---- KPIs
active = [i for i in incidents if i["status"] == "active"]
c1, c2, c3, c4, c5 = st.columns(5)
kpi(c1, "Flows analysed", f"{s['total']:,}")
kpi(c2, "Alerts", f"{s['attacks']:,}",
    f"{(s['attacks'] / s['total'] * 100) if s['total'] else 0:.1f}% of traffic", "#f97316")
kpi(c3, "Incidents", len(incidents), f"{len(active)} active", "#22d3ee")
kpi(c4, "Critical alerts", s["by_severity"].get("Critical", 0), "", SEV_COLORS["Critical"])
kpi(c5, "Analyst precision",
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
