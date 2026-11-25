---
title: FlowSight NIDS
emoji: 🛰️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 8501
pinned: false
short_description: Explainable ML network intrusion detection on CIC-IDS2017
---

# FlowSight: explainable network intrusion detection

A live demo of **FlowSight**, a machine-learning NIDS trained on the full CIC-IDS2017 dataset.
Real CIC-IDS2017 flows stream through the model continuously; every alert comes with a
SHAP-based, plain-English explanation, a severity and an incident.

* **Model:** XGBoost, macro-F1 **0.9979** on 377,499 held-out flows, 0.07% false positives,
  1.7 ms per flow
* **Classes:** Benign, DDoS, DoS, PortScan, BruteForce + Unknown-Anomaly
* **Pages:** Overview · Incidents · Investigate (what-if) · Models · Feedback · Analyse File ·
  Method & Limits (includes the final report)

This Space runs in **public-demo mode**: history can't be cleared, nothing is exported, and
uploaded files are classified but never stored.

Source code, final report and full experiment results:
https://github.com/gaurvit06/FLOWSIGHT
