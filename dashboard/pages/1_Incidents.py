"""Incidents (U3): alerts grouped into a timeline you can drill into."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st
from common import (
    PLOTLY_LAYOUT,
    SEV_COLORS,
    feedback_buttons,
    get,
    label_tag,
    sev_pill,
    shap_chart,
    story_card,
)
from common import setup as page_setup

page_setup("Incidents")

incidents = get("/incidents")
if not incidents:
    st.info("No incidents yet. Alerts are grouped into incidents as they arrive.")
    st.stop()

df = pd.DataFrame(incidents)
df["start"] = pd.to_datetime(df["start_ts"])
# a single-alert incident has zero duration; give it a few seconds so it's visible
df["end"] = [max(pd.to_datetime(e), s + timedelta(seconds=5))
             for s, e in zip(df["start"], df["end_ts"])]
df["name"] = df.apply(lambda r: f"#{r['id']} {r['label']}", axis=1)

fig = px.timeline(df, x_start="start", x_end="end", y="label", color="max_severity",
                  color_discrete_map=SEV_COLORS, hover_data=["id", "n_alerts", "status"],
                  title="Incident timeline")
fig.update_layout(**PLOTLY_LAYOUT, height=320, legend_title="severity", yaxis_title=None)
st.plotly_chart(fig, width="stretch")

c1, c2 = st.columns([1, 1.4])
with c1:
    st.subheader("All incidents")
    show = df[["id", "label", "max_severity", "n_alerts", "status", "start_ts"]].copy()
    show["start_ts"] = show["start_ts"].str[:19].str.replace("T", " ")
    st.dataframe(show.rename(columns={"max_severity": "severity", "n_alerts": "alerts",
                                      "start_ts": "started (UTC)"}),
                 hide_index=True, width="stretch", height=420)

with c2:
    pick = st.selectbox("Open incident", df["id"].tolist(),
                        format_func=lambda i: df.set_index("id").loc[i, "name"])
    inc = get(f"/incidents/{pick}")
    dur = timedelta(seconds=int(inc["duration_s"]))
    st.markdown(f"{sev_pill(inc['max_severity'])}{label_tag(inc['label'])} "
                f"**{inc['n_alerts']} alerts** over {dur} · {inc['status']}",
                unsafe_allow_html=True)
    st.markdown(f"> {inc['story']}")
    alerts = inc["alerts"]
    aid = st.selectbox("Alert", [a["id"] for a in alerts],
                       format_func=lambda i: f"#{i} · "
                       f"{next(a for a in alerts if a['id'] == i)['severity']}")
    a = next(x for x in alerts if x["id"] == aid)
    story_card(a)
    shap_chart(a["explanation"])
    feedback_buttons(a, key="inc")
    st.caption(f"Opened {datetime.fromisoformat(inc['start_ts']):%Y-%m-%d %H:%M:%S} UTC")
