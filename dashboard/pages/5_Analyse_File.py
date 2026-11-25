"""Upload a CSV of flows and classify all of them."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
from common import LABEL_COLORS, PLOTLY_LAYOUT, kpi, post
from common import setup as page_setup

page_setup("Analyse a file")

st.write("Upload a CSV of flows with the same columns as the training data. A `Label` "
         "column, if present, is used to show how many predictions match.")
f = st.file_uploader("Flows CSV", type="csv")
store = st.checkbox("Save results to history (shows up in Overview and Incidents)", value=False)

if f is not None:
    df = pd.read_csv(f, low_memory=False, encoding="latin-1")
    df.columns = df.columns.str.strip()
    truth = df.pop("Label") if "Label" in df else None
    df = df.head(5000).select_dtypes("number").replace([float("inf"), float("-inf")], None)
    flows = [{"features": {k: float(v) for k, v in r.items() if pd.notna(v)}}
             for r in df.to_dict("records")]
    with st.spinner(f"Classifying {len(flows)} flows..."):
        try:
            res = pd.DataFrame(post("/predict/batch", json={"flows": flows},
                                    params={"store": store}))
        except Exception as e:  # noqa: BLE001
            st.error(f"Prediction failed: {e}")
            st.stop()
    c1, c2, c3 = st.columns(3)
    kpi(c1, "Flows", len(res))
    kpi(c2, "Alerts", int(res["is_attack"].sum()), "", "#f97316")
    if truth is not None:
        from nids.data.clean import map_label
        t = truth.head(len(res)).map(map_label).values
        kpi(c3, "Matches ground truth", f"{(res['label'].values == t).mean():.1%}", "",
            "#22c55e")
        res.insert(0, "truth", t)
    fig = px.histogram(res, x="label", color="label", color_discrete_map=LABEL_COLORS)
    fig.update_layout(**PLOTLY_LAYOUT, height=300, showlegend=False)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(res[[c for c in ("truth", "label", "severity", "confidence", "story")
                      if c in res]], width="stretch")
