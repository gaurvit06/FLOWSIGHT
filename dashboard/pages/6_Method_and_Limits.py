"""How FlowSight works, where its data comes from, and what it can't do."""

from __future__ import annotations

import streamlit as st
from common import get
from common import setup as page_setup

page_setup("Method & limits")

model = get("/model")
reports = get("/reports")

st.subheader("How a flow becomes an alert")
st.graphviz_chart("""
digraph {
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#131a2e", color="#22d3ee",
        fontcolor="#e5e7eb", fontname="Helvetica", fontsize=11];
  edge [color="#64748b"];
  flow [label="Flow\\n(CICFlowMeter features)"];
  clean [label="Same cleaning\\nas training"];
  pipe [label="Scale → select k\\n→ classifier"];
  anom [label="Unknown-attack\\ndetector"];
  shap [label="SHAP\\ntop features"];
  story [label="Plain-English\\nstory"];
  sev [label="Severity\\n(type × confidence × burst)"];
  inc [label="Incident\\ngrouping"];
  db [label="History +\\nanalyst feedback", fillcolor="#0f2a33"];
  drift [label="Drift monitor\\n(PSI)", fillcolor="#0f2a33"];
  flow -> clean -> pipe -> shap -> story -> sev -> inc -> db;
  pipe -> anom [label=" if Benign", fontcolor="#94a3b8", fontsize=9];
  anom -> story;
  db -> drift;
}
""")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Training protocol")
    st.markdown(f"""
- **Data:** CIC-IDS2017 flow CSVs (see *Dataset audit* below).
- **Leakage controls:** IPs, source port, flow ID and timestamp removed; duplicates removed
  *before* splitting; scaling and feature selection fitted on training data only.
- **Split:** stratified 70 / 15 / 15 train / validation / test.
- **Model choice:** best **validation** macro-F1. The test set is used once, for the
  numbers you see.
- **Served model:** `{model['name']}` using {len(model['selected_features'])} of
  {len(model['features'])} features.
- **Unknown attacks:** Isolation Forest + range rule trained on benign flows only.
""")
with c2:
    st.subheader("Limits, stated honestly")
    st.markdown("""
- **Offline lab data.** CIC-IDS2017 was captured in 2017 on a testbed. Real networks differ,
  which is why the drift monitor and the cross-dataset report exist.
- **Flow statistics only.** No payload inspection, so it can't read encrypted content.
- **Detection, not prevention.** FlowSight never blocks traffic.
- **Shortcut features.** Some classes lean on destination port; an attacker can change it.
- **Adversarial evasion.** Attackers who know the features can shape their traffic.
- **Explanations ≠ proof.** SHAP shows what the model relied on, not intent.
- **Live capture** needs CICFlowMeter, and flows only exist once a connection ends
  (detection delay).
""")

st.divider()
DOCS = [
    ("FINAL_REPORT", "Final report", "the report is written by hand: reports/FINAL_REPORT.md"),
    ("DATASET", "Dataset audit & provenance", "python scripts/audit_dataset.py"),
    ("MODEL_CARD", "Model card", "python scripts/model_card.py"),
    ("port_ablation", "Destination Port ablation", "python scripts/port_ablation.py"),
]
for key, title, how in DOCS:
    with st.expander(title, expanded=False):
        if key in reports:
            # figure links in the reports are relative to reports/; they can't load here
            st.markdown("\n".join(line for line in reports[key].splitlines()
                                  if not line.startswith("![")))
        else:
            st.caption(f"Not generated yet: `{how}`.")
