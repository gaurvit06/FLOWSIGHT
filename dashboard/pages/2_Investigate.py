"""What-if explorer (U5): change a flow's most important features and watch the verdict move.

Sliders work on a log scale because network features span many orders of magnitude.
Predictions here use store=false, so experiments don't pollute the history.
"""

from __future__ import annotations

import math

import pandas as pd
import plotly.express as px
import streamlit as st
from common import LABEL_COLORS, PLOTLY_LAYOUT, get, label_tag, post, sev_pill
from common import setup as page_setup

page_setup("Investigate · what-if")

alerts = get("/predictions", limit=300, attacks_only=True)
if not alerts:
    st.info("No alerts to investigate yet.")
    st.stop()

model = get("/model")
baseline = get("/baseline")


def to_log(x: float, lo: float) -> float:
    """Slider position for value x (log scale, shifted so the minimum maps to 0)."""
    return math.log10(max(x - lo, 0) + 1)


by_id = {a["id"]: a for a in alerts}
aid = st.selectbox("Alert to investigate", list(by_id),
                   format_func=lambda i: f"#{i} · {by_id[i]['label']} · {by_id[i]['severity']}")
alert = get(f"/predictions/{aid}")
features = dict(alert["features"])

top = [e["feature"] for e in alert["explanation"]][:5] or model["selected_features"][:5]
st.caption("Drag the sliders to change this flow. Everything else stays as recorded. "
           "Ask: *how much would this have to change before the model calls it benign?*")

left, right = st.columns([1, 1.2])
with left:
    st.subheader("Top features")
    for f in top:
        v = float(features[f])
        b = baseline.get(f, {})
        lo = min(b.get("min", v), v, 0.0)
        hi = max(b.get("max", v), v) * 1.5 + 1
        pos = st.slider(f, 0.0, round(to_log(hi, lo), 3), round(to_log(v, lo), 3), step=0.01,
                        key=f"s{aid}{f}",
                        help=f"normal median {b.get('p50', float('nan')):,.2f} · "
                             f"normal p99 {b.get('p99', float('nan')):,.2f}")
        features[f] = 10 ** pos - 1 + lo
        st.caption(f"value **{features[f]:,.2f}**  (recorded {v:,.2f}; normal median "
                   f"{b.get('p50', float('nan')):,.2f})")
    if st.button("Reset"):
        for f in top:
            st.session_state.pop(f"s{aid}{f}", None)
        st.rerun()

with right:
    res = post("/predict", json={"features": features}, params={"store": False})
    changed = res["label"] != alert["label"]
    st.subheader("Model verdict")
    st.markdown(f"{sev_pill(res.get('severity'))}{label_tag(res['label'])} "
                f"**{res['confidence']:.0%}** confident"
                + ("  ← changed from " + alert["label"] if changed else ""),
                unsafe_allow_html=True)
    st.markdown(f"> {res['story']}")
    probs = pd.DataFrame(res["probabilities"].items(), columns=["class", "p"])
    fig = px.bar(probs, x="p", y="class", orientation="h", color="class",
                 color_discrete_map=LABEL_COLORS, range_x=[0, 1], title="Class probabilities")
    fig.update_layout(**PLOTLY_LAYOUT, height=300, showlegend=False, yaxis_title=None)
    st.plotly_chart(fig, width="stretch")
    if res.get("anomaly_score") is not None:
        st.caption(f"Unknown-attack detector score: {res['anomaly_score']:+.3f} "
                   "(> 0 means unlike normal traffic)")
