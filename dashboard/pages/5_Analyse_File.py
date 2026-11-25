"""Upload a CSV of flows, classify all of them, triage by confidence, download the results."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from common import LABEL_COLORS, PLOTLY_LAYOUT, kpi, post
from common import setup as page_setup

page_setup("Analyse a file")

st.write("Upload a CSV of flows with the same columns as the training data (for example a "
         "CICIDS2017 file or CICFlowMeter output). A `Label` column, if present, is used to "
         "show how many predictions match.")
f = st.file_uploader("Flows CSV", type="csv")
c1, c2 = st.columns(2)
max_rows = c1.number_input("Rows to analyse", 100, 50_000, 5_000, step=1_000)
store = c2.checkbox("Save results to history (Overview & Incidents)", value=False)


@st.cache_data(show_spinner=False)
def classify(data: bytes, n: int, store: bool) -> pd.DataFrame:
    import io

    df = pd.read_csv(io.BytesIO(data), low_memory=False, encoding="latin-1", nrows=n)
    df.columns = df.columns.str.strip()
    truth = df.pop("Label") if "Label" in df else None
    num = df.select_dtypes("number").replace([np.inf, -np.inf], np.nan)
    keep = num.notna().all(axis=1)              # same rule as training: drop inf/NaN rows
    num, truth = num[keep], (truth[keep] if truth is not None else None)
    preds = []
    for start in range(0, len(num), 1000):      # batches keep requests small
        chunk = num.iloc[start:start + 1000]
        preds += post("/predict/batch", params={"store": store},
                      json={"flows": [{"features": r} for r in chunk.to_dict("records")]})
    res = pd.DataFrame(preds)
    res.insert(0, "row", num.index.to_numpy() + 2)     # line number in the CSV (1 = header)
    if truth is not None:
        from nids.data.clean import map_label
        res.insert(1, "truth", truth.map(map_label).to_numpy())
    return res


if f is not None:
    with st.spinner("Classifying..."):
        try:
            res = classify(f.getvalue(), int(max_rows), store)
        except Exception as e:  # noqa: BLE001
            st.error(f"Prediction failed: {e}")
            st.stop()

    # ---- confidence triage
    thr = st.slider("Confidence threshold: predictions below this are marked **needs review**",
                    0.5, 0.99, 0.80, 0.01)
    res["needs_review"] = res["confidence"] < thr

    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, "Flows", f"{len(res):,}")
    kpi(k2, "Alerts", f"{int(res['is_attack'].sum()):,}", "", "#f97316")
    kpi(k3, "Needs review", f"{int(res['needs_review'].sum()):,}",
        f"confidence < {thr:.0%}", "#eab308")
    if "truth" in res:
        match = (res["label"] == res["truth"]).mean()
        kpi(k4, "Matches ground truth", f"{match:.1%}", "", "#22c55e")

    left, right = st.columns(2)
    fig = px.histogram(res, x="label", color="label", color_discrete_map=LABEL_COLORS,
                       title="Verdicts")
    fig.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False)
    left.plotly_chart(fig, width="stretch")
    fig = px.histogram(res, x="confidence", color="label", nbins=30, log_y=True,
                       color_discrete_map=LABEL_COLORS, title="Confidence distribution")
    fig.add_vline(x=thr, line_dash="dash", line_color="#eab308",
                  annotation_text="threshold", annotation_font_color="#eab308")
    fig.update_layout(**PLOTLY_LAYOUT, height=320, legend_title=None)
    right.plotly_chart(fig, width="stretch")

    view = st.radio("Show", ["Alerts", "Needs review", "Mismatches", "All"], horizontal=True)
    shown = {"Alerts": res[res["is_attack"]], "Needs review": res[res["needs_review"]],
             "Mismatches": res[res["label"] != res["truth"]] if "truth" in res else res.iloc[0:0],
             "All": res}[view]
    cols = [c for c in ("row", "truth", "label", "severity", "confidence", "needs_review",
                        "story") if c in shown]
    st.dataframe(shown[cols], width="stretch", hide_index=True)

    export = res.drop(columns=[c for c in ("explanation", "probabilities") if c in res])
    st.download_button("⬇ Download annotated CSV", export.to_csv(index=False).encode(),
                       file_name=f"flowsight_{f.name}", mime="text/csv")
