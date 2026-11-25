"""Model performance, attack fingerprints (U9) and experiment reports (U8)."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from common import LABEL_COLORS, PLOTLY_LAYOUT, get, kpi
from common import setup as page_setup

page_setup("Models")

data = get("/models")
models = data["models"]

rows = []
for m in models:
    mt = m["metrics"]
    rows.append({"model": m["name"] + ("  ★ serving" if m["best"] else ""),
                 "macro F1": mt["macro_f1"], "accuracy": mt["accuracy"],
                 "ROC-AUC": mt["roc_auc_ovr"], "false positive rate": mt["false_positive_rate"],
                 "latency ms/flow": mt["latency_ms_per_flow"],
                 "train s": mt.get("train_seconds")})
st.subheader("Model comparison")
st.dataframe(pd.DataFrame(rows).round(4), width="stretch", hide_index=True)
st.caption("Picked by macro-F1, which weights every attack class equally. Accuracy is "
           "misleading when most traffic is benign.")

if data.get("anomaly"):
    am = data["anomaly"]["metrics"]
    rec = am.get("missed_attacks_recovered")
    c1, c2, c3 = st.columns(3)
    kpi(c1, "Missed attacks recovered", "–" if rec is None else f"{rec:.0%}",
        f"of {am.get('classifier_missed_attacks', '?')} the classifier called Benign",
        "#22d3ee")
    kpi(c2, "Extra benign false alarms", f"{am.get('rule_benign_false_alarm_rate', 0):.2%}",
        "cost of the unknown-attack rule", "#eab308")
    kpi(c3, "Isolation Forest alone", f"{am['attack_flag_rate']:.1%}",
        "of attacks flagged (weak on this dataset)", "#94a3b8")

st.divider()
name = st.selectbox("Details for", [m["name"] for m in models])
m = next(x for x in models if x["name"] == name)
c1, c2 = st.columns(2)
fig = px.imshow(m["metrics"]["confusion_matrix"], x=m["classes"], y=m["classes"],
                text_auto=True, color_continuous_scale="Teal",
                labels={"x": "Predicted", "y": "Actual"}, title="Confusion matrix")
fig.update_layout(**PLOTLY_LAYOUT, height=420)
c1.plotly_chart(fig, width="stretch")
pc = pd.DataFrame(m["metrics"]["per_class"]).T.reset_index(names="class")
fig = px.bar(pc.melt(id_vars="class", value_vars=["precision", "recall", "f1-score"]),
             x="class", y="value", color="variable", barmode="group", range_y=[0, 1.05],
             title="Per-class metrics",
             color_discrete_sequence=["#38bdf8", "#22d3ee", "#a78bfa"])
fig.update_layout(**PLOTLY_LAYOUT, height=420, legend_title=None)
c2.plotly_chart(fig, width="stretch")
st.write("**Selected features:** " + ", ".join(m["selected_features"]))

# ---- Attack fingerprints (U9)
fp = data.get("fingerprints") or {}
if fp.get("classes"):
    st.divider()
    st.subheader("Attack fingerprints")
    st.caption("Median value of the top features for each class, scaled 0–1 across all "
               "traffic. Each attack has a recognisable shape.")
    chosen = st.multiselect("Classes", list(fp["classes"]), default=list(fp["classes"]))
    radar = go.Figure()
    theta = fp["features"] + fp["features"][:1]
    for cls in chosen:
        vals = fp["classes"][cls]
        radar.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=theta, name=cls,
                                        fill="toself", opacity=0.45,
                                        line_color=LABEL_COLORS.get(cls)))
    radar.update_layout(**PLOTLY_LAYOUT, height=520,
                        polar={"bgcolor": "rgba(0,0,0,0)",
                                   "radialaxis": {"range": [0, 1], "showticklabels": False}})
    st.plotly_chart(radar, width="stretch")

# ---- Experiment reports (U8 etc.)
reports = get("/reports")
if reports:
    st.divider()
    st.subheader("Experiment reports")
    for key, title in [("results", "Cross-validated results"),
                       ("zero_day", "Zero-day (leave-one-attack-out)"),
                       ("cross_dataset", "Generalization report card")]:
        if key in reports:
            with st.expander(title, expanded=key == "cross_dataset"):
                st.markdown(reports[key])
else:
    st.caption("Run the scripts in `scripts/` to add cross-validation, zero-day and "
               "cross-dataset reports here.")
