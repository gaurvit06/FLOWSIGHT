# BUILD.md: from scaffold to a finished, distinctive NIDS

This is the build plan for the project. It starts from the scaffold already in this repo and
ends with a system you can demo, defend in a viva, and put on a CV.

**Working name:** *FlowSight*, an intrusion detector that explains itself in plain English.

---

## 1. What makes this project different

Most student NIDS projects stop at "trained 3 models, here's a confusion matrix." This one is
built around one idea:

> **An alert is only useful if an analyst can understand it, trust it, and act on it.**

Every unique feature below follows from that idea:

| # | Feature | Typical NIDS project | FlowSight |
|---|---|---|---|
| U1 | **Plain-English alert stories** | Shows a label | *"Flagged as DDoS: 48,000 packets/s, 900× the normal median, all tiny 60-byte SYN packets"* |
| U2 | **Severity scoring** | Every alert is the same | Low / Medium / High / Critical from attack type × confidence × burst rate |
| U3 | **Incident grouping** | 5,000 separate alerts | Alerts grouped into a few *incidents* on a timeline |
| U4 | **Unknown-attack detector** | Can only detect trained classes | An anomaly model flags *"looks abnormal, but not like any known attack"* |
| U5 | **What-if explorer** | Static charts | Move sliders on a flow's features and watch the prediction change |
| U6 | **Analyst feedback loop** | No feedback | Mark alerts as *confirmed* or *false positive*, track real-world precision, feed into retraining |
| U7 | **Drift monitor** | Assumes data never changes | Warns when live traffic stops looking like the training data |
| U8 | **Generalization report card** | Reports 99.9% on one dataset | Trains on CICIDS2017, tests on CSE-CIC-IDS2018, reports the honest drop |
| U9 | **Attack fingerprints** | — | A radar chart of each attack class's behavioural profile |

You don't need all nine. **U1, U2, U4 and U8** give the most for the effort. Build them first.

---

## 2. Build status

Everything that can be built without the real datasets is **done and tested** (45 tests).
What remains is work only you can do: download the data, run the experiments on it, make
the judgement calls, and write the report.

| Component | Status | Where |
|---|---|---|
| Cleaning, shared pipeline, RF/XGB/SVM training, metrics, registry | ✅ done | `src/nids/` |
| Per-flow SHAP explanations | ✅ done | `src/nids/explain/shap_explainer.py` |
| U1 Plain-English alert stories | ✅ done | `src/nids/explain/narrative.py` |
| U2 Severity scoring | ✅ done | `src/nids/scoring/`, `configs/severity.yaml` |
| U3 Incident grouping + timeline | ✅ done | `src/nids/storage/incidents.py`, Incidents page |
| U4 Unknown-attack detector + zero-day experiment | ✅ done | `src/nids/models/anomaly.py`, `scripts/zero_day_experiment.py` |
| U5 What-if explorer | ✅ done | `dashboard/pages/2_Investigate.py` |
| U6 Analyst feedback loop + export | ✅ done | `src/nids/storage/feedback.py`, Feedback page |
| U7 Drift monitor (PSI) + banner | ✅ done | `src/nids/monitoring/drift.py`, Overview page |
| U8 Generalization report card | ✅ script done | `scripts/cross_dataset.py`, needs the 2018 data |
| U9 Attack fingerprints | ✅ done | Models page |
| CV, tuning, feature-selection study, imbalance ablation, global SHAP | ✅ scripts done | `scripts/`, `src/nids/models/tune.py` |
| EDA notebook | ✅ done | `notebooks/01_eda.ipynb` |
| Live capture bridge | ✅ code done, untested | `src/nids/capture/live.py`, needs CICFlowMeter + VM lab |
| Docker, CI, pinned deps | ✅ written | Docker image build not yet tested |
| **Real-data results, decisions, report** | ⬜ **you** | See the unticked boxes below |

## 3. Setup

```powershell
python -m uv venv --python 3.12 .venv
python -m uv pip install --python .venv\Scripts\python.exe -e ".[dev]"
.\.venv\Scripts\Activate.ps1
python scripts/make_sample_data.py      # synthetic data, so everything runs today
python -m nids.models.train
pytest
```

> **Python version:** use **3.11 or 3.12**. Very new Python releases sometimes lack prebuilt
> wheels for `shap`/`numba` or `xgboost`, and installation then fails or compiles from source.

Then switch to the real data (see README → *Real dataset*). Set `sample_frac: 0.1` while you
iterate, and use `1.0` only for final numbers.

---

## 4. Build phases

Each phase has **tasks**, the **files** it touches, and a **"done when"** check. Don't start a
phase until the previous one meets its "done when".

### Phase 1: Real data and EDA (Week 1)

- [ ] Download CICIDS2017 into `data/raw/` and delete the synthetic CSV
- [x] `notebooks/01_eda.ipynb`: class balance, missing/inf counts, duplicate rate, feature
      correlation heatmap, distributions of 5 key features per class
- [ ] Record in the notebook how many rows cleaning removes and why
- [ ] Decide whether to keep `Destination Port` as a feature. It helps catch brute force
      (22/21), but the model may then just memorise ports. Try both and record the result.

**Done when:** `python -m nids.models.train --rebuild` runs on real data and you can explain
every number in the cleaning log.

### Phase 2: Proper experiments (Week 2)

- [x] Replace the single split with **stratified 5-fold cross-validation** for model selection.
      Keep a held-out test set that you touch **once**, at the end.
- [x] Hyperparameter search (`optuna` or `RandomizedSearchCV`) for RF and XGBoost
- [x] `scripts/feature_selection_study.py`: macro-F1 vs. k ∈ {5, 10, 15, 20, 30, 40, all} for
      `f_classif`, `mutual_info` and model importance. Save a plot to `reports/figures/`.
- [x] Ablation: SMOTE vs. class weights vs. nothing, using per-class recall
- [x] Global SHAP summary plot per class to `reports/figures/`

**Files:** `src/nids/models/train.py`, new `src/nids/models/tune.py`, `scripts/`

**Done when:** `reports/results.md` has a results table with mean ± std over folds for all
three models.

### Phase 3: Plain-English alert stories (U1)

The SHAP output says *which* features mattered. A story says *why*, by comparing the flow's
values to **normal (benign) traffic**.

**Design**
1. At training time, save per-feature percentiles of the **benign** class
   (`p1`…`p99`) to `models/profiles/baseline.json`.
2. For each alert, take the top 3 SHAP features. For each one, compute
   `ratio = value / benign_p50` and a percentile position.
3. Render the result with templates. No LLM is needed, so it's deterministic and testable.

```python
# src/nids/explain/narrative.py (sketch)
def describe(feature, value, base):
    if value > base["p99"]:
        ratio = value / max(base["p50"], 1e-9)
        return f"{feature} is {value:,.0f}, {ratio:,.0f}× the normal median (above 99% of normal traffic)"
    if value < base["p1"]:
        return f"{feature} is unusually low ({value:,.2f}; normal median {base['p50']:,.2f})"
    return f"{feature} = {value:,.2f}"

def story(label, confidence, top_features, baseline):
    reasons = "; ".join(describe(f["feature"], f["value"], baseline[f["feature"]])
                        for f in top_features[:3])
    return f"Flagged as {label} ({confidence:.0%} confident) because {reasons}."
```

- [x] Save `baseline.json` in `train.py`
- [x] `src/nids/explain/narrative.py` and tests
- [x] Add a `story` field to the `Prediction` schema and show it as the alert headline in
      the dashboard

**Done when:** every alert in the dashboard has a sentence a non-ML person can follow.

### Phase 4: Severity scoring (U2)

```
severity_score = base_severity[label] × confidence × burst_factor
```

| Attack | Base severity | Why |
|---|---|---|
| DDoS | 0.9 | Takes the service offline |
| DoS | 0.8 | Same, from a single source |
| BruteForce | 0.7 | Can lead to account takeover |
| PortScan | 0.4 | Reconnaissance, often comes before an attack |
| Unknown anomaly (U4) | 0.6 | No known type, so it needs a human look |

`burst_factor` = 1 + log10(number of same-type alerts in the last 60 s). Map the score to
**Low < 0.3 ≤ Medium < 0.55 ≤ High < 0.75 ≤ Critical**. Keep the weights in
`configs/severity.yaml`, not in code.

- [x] `src/nids/scoring/severity.py`, the config file, and tests
- [x] Add `severity` to the DB table and API response. Colour-code alerts in the dashboard.

### Phase 5: Unknown-attack detector (U4)

A classifier can only output classes it has seen. To catch new attacks:

1. Train an **Isolation Forest** on **benign training flows only** (`contamination="auto"`).
2. At prediction time: if the classifier says *Benign* **but** the anomaly score is in the top
   0.5% of the benign training scores, label the flow **`Unknown-Anomaly`**.
   *Built:* a **range rule** was added too (any feature >10× beyond anything benign),
   because the forest averages over all features and can dilute one extreme value.
3. **Evaluate it honestly:** retrain the classifier *without* one attack class (e.g. PortScan),
   then measure how many PortScan flows the detector catches. This leave-one-attack-out
   experiment is a strong result for your report.

- [x] `src/nids/models/anomaly.py`, saved as `models/anomaly/`
- [x] `scripts/zero_day_experiment.py` that runs leave-one-attack-out for each class and
      writes a table

### Phase 6: Incident grouping (U3)

Group alerts into incidents: same attack type, gap between alerts under 120 s. Once live
capture exists, also group by the same target. An incident has a start time, end time, flow
count, highest severity, and a story built from its strongest alert.

- [x] `src/nids/storage/incidents.py` (SQL `incidents` table, updated on each new alert)
- [x] `/incidents` endpoint
- [x] Dashboard **Incidents** tab: a Gantt-style timeline (`plotly.express.timeline`), one bar
      per incident, coloured by severity

### Phase 7: Dashboard upgrade (U5, U6, U9)

Rework the dashboard into five pages:

| Page | Contents |
|---|---|
| **Overview** | KPI tiles (flows, alerts, open incidents, current FPR), live traffic chart, drift banner (U7) |
| **Incidents** | Timeline, then click through to its alerts, then each alert's story and SHAP chart |
| **Investigate** | **What-if explorer (U5):** pick an alert, its top 5 features appear as sliders, and the prediction and probabilities update live via `/predict?store=false`. Shows *"how much would this have to change to look benign?"* |
| **Models** | Comparison table, confusion matrices, feature-selection curve, **attack fingerprint radar (U9):** median normalised value of the top 8 features per class |
| **Feedback** | **(U6)** Each alert has ✅ Confirmed / ❌ False positive buttons stored in a `feedback` table. Shows precision measured from analyst labels, and exports labelled flows to `data/feedback/` for retraining. |

- [x] Split `dashboard/Overview.py` into `dashboard/pages/*.py` (Streamlit multipage)
- [x] New endpoints: `POST /feedback/{id}`, `GET /feedback/summary`, `GET /incidents`
- [x] Pick a consistent theme in `.streamlit/config.toml`, plus one severity colour scale used
      everywhere

### Phase 8: Drift monitor (U7)

- [x] Save training feature histograms (10 bins each) to `models/<name>/reference.json`
- [x] Compute the **Population Stability Index (PSI)** over the last N stored flows for each
      selected feature. Flag a feature when PSI > 0.2.
- [x] `/drift` endpoint, and a banner on the Overview page:
      *"3 features have drifted. Consider retraining."*
- [ ] Demo it: replay CSE-CIC-IDS2018 flows into a model trained on 2017 and watch the banner
      appear.

### Phase 9: Generalization report card (U8)

- [x] `scripts/cross_dataset.py`: train on CICIDS2017 and evaluate on CSE-CIC-IDS2018 (map the
      columns. Most share names, but check). Report per-class recall in both datasets side by side.
- [ ] Write 1–2 paragraphs on **why** it drops: different network, tools and feature-extractor
      version. Examiners respect this far more than a perfect score.

### Phase 10: Live capture (stretch goal)

- [x] `src/nids/capture/live.py`: CICFlowMeter (or a Scapy flow aggregator) → same features →
      `/predict`
- [ ] **Lab only:** two VMs on a host-only network (e.g. Ubuntu target and Kali attacker).
      Generate traffic with `nmap`, `hping3` and `hydra` **against your own VM only**. Never
      against networks you don't own or have written permission to test.
- [ ] Document how accuracy on live traffic differs from the dataset. That's a valid result too.

### Phase 11: Packaging and polish

- [x] `Dockerfile` and `docker-compose.yml` (api, dashboard; the SQLite volume)
- [x] GitHub Actions: `ruff` and `pytest` on the synthetic data for every push
- [x] Pin dependency versions (`pip freeze > requirements.lock`)
- [ ] README: architecture diagram, results table, 3 dashboard screenshots, a GIF of a replay
- [x] Remove dead code; every module gets a docstring

---

## 5. Final repository shape

```
nids/
├── BUILD.md  README.md  pyproject.toml  Makefile  Dockerfile  docker-compose.yml
├── configs/        train.yaml  severity.yaml
├── data/           raw/  processed/  feedback/
├── models/         rf/ xgb/ svm/ anomaly/  best.json
├── notebooks/      01_eda.ipynb  02_shap_global.ipynb
├── reports/        results.md  figures/
├── scripts/        make_sample_data.py  feature_selection_study.py
│                   zero_day_experiment.py  cross_dataset.py
├── src/nids/
│   ├── data/       load.py  clean.py
│   ├── features/   pipeline.py
│   ├── models/     factory.py  train.py  tune.py  evaluate.py  registry.py  anomaly.py
│   ├── explain/    shap_explainer.py  narrative.py
│   ├── scoring/    severity.py
│   ├── monitoring/ drift.py
│   ├── storage/    db.py  incidents.py  feedback.py
│   ├── capture/    replay.py  live.py
│   └── api/        main.py  schemas.py
├── dashboard/      Overview.py  common.py  pages/   (theme: .streamlit/config.toml)
└── tests/
```

---

## 6. Testing strategy

| Layer | Test | Why |
|---|---|---|
| Data | Cleaning invariants (no inf/NaN, no leaky columns, no duplicates) | Leakage bugs are silent |
| Features | Pipeline output shape = k; same input gives same output | Training and serving must match |
| Explain | Story mentions the top SHAP feature; ratios are correct for fixed inputs | Templates break quietly |
| Scoring | Known inputs give the expected severity band | Config changes shouldn't surprise |
| API | 422 on bad input; attack responses include story, severity and explanation | Contract for the dashboard |
| Golden flows | 10 hand-picked real flows keep their label after retraining | Catches regressions |
| Quality gate | macro-F1 and per-attack recall ≥ thresholds | A worse model breaks the build |

---

## 7. Demo script (5 minutes, for a viva or interview)

1. **Problem (30 s):** rule-based IDS misses new attacks, and ML IDS are black boxes.
2. **Models (60 s):** Models page, with the comparison table and the honest generalization
   report card.
3. **Live (90 s):** start `replay` with attacks mixed in. Incidents appear on the timeline,
   severity colours change, and the drift banner stays green.
4. **Explain (60 s):** open a Critical DDoS incident, read the plain-English story aloud, show
   the SHAP bars.
5. **What-if (30 s):** drag packets/s down until the prediction becomes Benign. *"This is the
   model's decision boundary."*
6. **Unknown attack (30 s):** show the leave-one-attack-out result. The anomaly detector catches
   an attack the classifier never saw.

---

## 8. Report checklist

- [ ] Dataset description, cleaning steps, and row counts before and after
- [ ] Why accuracy is misleading here; the metrics you chose and why
- [ ] Model comparison with mean ± std over cross-validation folds
- [ ] Feature selection study (plot and chosen k)
- [ ] Global and local SHAP findings: do the top features make sense for each attack?
- [ ] Zero-day (leave-one-attack-out) results
- [ ] Cross-dataset generalization and why it drops
- [ ] Limitations: dataset age, encrypted traffic, adversarial evasion, lab vs. production
- [ ] Future work

---

## 9. Timeline (solo, about 6 weeks)

| Week | Phases | Milestone |
|---|---|---|
| 1 | 1 | Real data cleaned, EDA notebook |
| 2 | 2 | Cross-validated results table and figures |
| 3 | 3, 4 | Alerts have stories and severity |
| 4 | 5, 6 | Unknown-attack detector and incidents |
| 5 | 7, 8 | Final dashboard with what-if, feedback and drift |
| 6 | 9, 11 (+10 if time) | Generalization report, Docker, CI, README, demo rehearsal |

---

## 10. Pitfalls to avoid

1. **Results of 99.9% or more:** check for duplicates across train and test, leaky columns, and
   that test data wasn't used for feature selection.
2. **Tuning on the test set:** use cross-validation for every decision. Touch the test set once.
3. **Fitting SMOTE before splitting:** synthetic copies of test rows leak into training.
4. **Different features at train and serve time:** always go through the saved pipeline.
5. **Explaining with scaled values:** stories must show raw values (packets/s, bytes), not
   z-scores. The scaffold already does this.
6. **Pointing attack tools at real networks:** lab VMs only.
