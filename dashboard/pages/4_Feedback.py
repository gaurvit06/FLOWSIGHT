"""Analyst feedback (U6): review alerts, measure real-world precision, export for retraining."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from common import feedback_buttons, get, kpi, post, story_card
from common import setup as page_setup

page_setup("Analyst feedback")

fb = get("/feedback/summary")
c1, c2, c3, c4 = st.columns(4)
kpi(c1, "Precision (analyst-measured)",
    f"{fb['precision']:.0%}" if fb["precision"] is not None else "–", "", "#22c55e")
kpi(c2, "Confirmed", fb["confirmed"], "", "#22c55e")
kpi(c3, "False positives", fb["false_positive"], "", "#ef4444")
kpi(c4, "Waiting for review", fb["unreviewed"], "", "#eab308")

if fb["per_label"]:
    st.write("")
    st.dataframe(pd.DataFrame(fb["per_label"]).T.reset_index(names="attack type"),
                 hide_index=True, width="stretch")

st.divider()
left, right = st.columns([2, 1])
with left:
    st.subheader("Review queue")
    queue = [a for a in get("/predictions", limit=300, attacks_only=True) if not a.get("verdict")]
    if not queue:
        st.success("Nothing waiting for review.")
    for a in queue[:15]:
        story_card(a)
        feedback_buttons(a, key="q")
with right:
    st.subheader("Retraining data")
    st.write("Reviewed flows are exported with corrected labels (false positive → Benign). "
             "Add the file to `data/raw/` to include it in the next training run.")
    if st.button("Export labelled flows", width="stretch"):
        r = post("/feedback/export")
        st.success(f"Exported {r['rows']} rows to `{r['path']}`") if r["rows"] else \
            st.warning("Nothing reviewed yet.")
