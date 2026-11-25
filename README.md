# FlowSight: an intrusion detector that explains itself

FlowSight is a machine-learning **Network Intrusion Detection System**. It classifies network
flows as **Benign, DDoS, DoS, PortScan or BruteForce** with Random Forest, XGBoost and a linear
SVM. It also catches **attacks it was never trained on**, and explains every alert in **plain
English**.

> *"Flagged as DDoS (100% confident): packet rate was 53,220 packets/s, 1,061× the normal
> median (higher than 99% of normal traffic); and packets received was unusually low at 0."*

```
flows (replay / CSV / live CICFlowMeter)
        │
        ▼
FastAPI ─► classifier (scale → select k → RF/XGB/SVM)
        ├► unknown-attack detector (Isolation Forest + range rule, benign-only)
        ├► SHAP explanation ─► plain-English story
        ├► severity score (type × confidence × burst)
        ├► SQLite: predictions · incidents · analyst feedback
        └► drift monitor (PSI vs training data)
                 │
                 ▼
Streamlit dashboard: Overview · Incidents · Investigate (what-if) · Models · Feedback · Analyse File
```

**📄 Final report:** [reports/FINAL_REPORT.md](reports/FINAL_REPORT.md). Results on the full
CIC-IDS2017 dataset: XGBoost **macro-F1 0.9979** on 377,499 held-out flows, 0.07% false
positives, 1.7 ms per flow, plus honest zero-day and robustness findings.

## Public demo

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_demo.ps1     # prints a public https URL
powershell -ExecutionPolicy Bypass -File scripts\stop_demo.ps1    # stops everything
```

The launcher streams real CIC-IDS2017 flows through the API and publishes **only the
dashboard** through a free serveo.net SSH tunnel (no install needed). It runs in demo mode:
history can't be cleared, nothing is exported to disk, and uploads are limited to 20 MB and
5,000 rows. The API stays on localhost. The URL works only while your computer runs the demo,
and visitors see a one-time serveo warning page first.

## What makes it different

| | Feature | Where |
|---|---|---|
| U1 | **Plain-English alert stories**, comparing each flow to normal traffic | `src/nids/explain/narrative.py` |
| U2 | **Severity scoring**: Low / Medium / High / Critical | `src/nids/scoring/`, `configs/severity.yaml` |
| U3 | **Incident grouping**: thousands of alerts become a timeline of incidents | `src/nids/storage/incidents.py` |
| U4 | **Unknown-attack detector**: catches attacks the classifier never saw | `src/nids/models/anomaly.py` |
| U5 | **What-if explorer**: drag a flow's features and watch the verdict change | `dashboard/pages/2_Investigate.py` |
| U6 | **Analyst feedback loop**: confirm or reject alerts, measure real precision, export for retraining | `src/nids/storage/feedback.py` |
| U7 | **Drift monitor**: warns when traffic stops looking like the training data | `src/nids/monitoring/drift.py` |
| U8 | **Generalization report card**: an honest cross-dataset evaluation | `scripts/cross_dataset.py` |
| U9 | **Attack fingerprints**: a radar chart of each attack's behaviour | Models page |

## Quick start (Windows PowerShell)

Use **Python 3.12**. Newer versions may lack prebuilt packages for `shap`/`xgboost`.
[uv](https://docs.astral.sh/uv/) installs much faster than pip (`pip install uv`).

```powershell
python -m uv venv --python 3.12 .venv
python -m uv pip install --python .venv\Scripts\python.exe -e ".[dev]"
.\.venv\Scripts\Activate.ps1

python scripts/make_sample_data.py      # synthetic data so everything runs today
python -m nids.models.train             # trains RF/XGB/SVM + anomaly detector + profiles
pytest                                  # 49 tests, including a model quality gate
```

Then use three terminals:

```powershell
uvicorn nids.api.main:app --port 8000                 # API, docs at http://localhost:8000/docs
streamlit run dashboard/Overview.py                   # dashboard at http://localhost:8501
python -m nids.capture.replay --n 500 --delay 0.05    # stream traffic through it
```

Dashboard pages: **Overview** (KPIs, drift banner, top alerts) · **Incidents** (timeline,
drill-down) · **Investigate** (what-if sliders) · **Models** (comparison, fingerprints,
reports) · **Feedback** (review queue, precision, export) · **Analyse File** (upload a CSV,
confidence threshold, "needs review" triage, annotated CSV download) · **Method & Limits**
(pipeline diagram, training protocol, limitations, dataset audit, model card).

Or with Docker (after training on the host): `docker compose up --build`.

## Experiments (for the report)

Every script writes a Markdown report to `reports/` (shown on the dashboard) and figures to
`reports/figures/`. Data is split **70 / 15 / 15** (train / validation / test). Every
modelling decision, including which model is served, uses the training or validation data.
The test set is used once, for the final numbers. See [DECISIONS.md](DECISIONS.md) for why.

| Script | What it answers |
|---|---|
| `scripts/fetch_dataset.py --zip <file>` | Extract CIC-IDS2017 into `data/raw/`, record SHA-256 checksums, remove synthetic data |
| `scripts/audit_dataset.py` | `reports/DATASET.md`: provenance, per-file rows/labels/inf/NaN/duplicates, cleaning log, class balance |
| `scripts/model_card.py` | `reports/MODEL_CARD.md`: intended use, data, split, per-class results, limitations |
| `scripts/cv_report.py [--tune]` | Stratified 5-fold CV, mean ± std for every model (+ random hyperparameter search) |
| `scripts/feature_selection_study.py` | Macro-F1 vs. number of features for ANOVA F, mutual information and RF importance |
| `scripts/imbalance_ablation.py` | Nothing vs. class weights vs. SMOTE (SMOTE applied inside training only) |
| `scripts/shap_global.py` | Global SHAP importance per attack class |
| `scripts/zero_day_experiment.py` | Leave-one-attack-out: how much the unknown-attack detector catches |
| `scripts/cross_dataset.py --test-dir <dir>` | Train on one dataset, test on another, and report the drop |
| `notebooks/01_eda.ipynb` | Class balance, data quality, cleaning log, correlations, the Destination Port question |

Use `--max-rows` to subsample the training data for speed (default 200,000).

## Real datasets

1. **CIC-IDS2017**: on unb.ca/cic/datasets/ids-2017.html choose *Download this dataset*,
   fill in the short registration form, and download `CSVs/MachineLearningCSV.zip` (~230 MB).
2. `python scripts/fetch_dataset.py --zip <path to MachineLearningCSV.zip>`
   (extracts the 8 CSVs, writes checksums, removes synthetic data and stale caches)
3. `python scripts/audit_dataset.py` → `reports/DATASET.md`
4. `python -m nids.models.train --rebuild` (training rows are capped by `max_train_rows` in
   `configs/train.yaml` to fit in memory; validation and test stay full size)
5. `python scripts/model_card.py`, then the experiment scripts above.
6. For the generalization report card, put **CSE-CIC-IDS2018** CSVs in e.g.
   `data/cse-cic-ids2018/` and run
   `python scripts/cross_dataset.py --test-dir data/cse-cic-ids2018 --name "CSE-CIC-IDS2018"`.
   Column names that differ are mapped in `src/nids/data/columns.py`.

> **The synthetic data is made up.** Its near-perfect scores only prove the code works end to
> end. Never put them in your report.

## Live capture (lab only)

`python -m nids.capture.live --watch data/live` sends every flow CICFlowMeter writes into that
folder to the API. Generate attack traffic **only between VMs you own on an isolated host-only
network**.

## Design decisions

- **One pipeline for training and serving.** Scaling and feature selection are fitted inside
  the saved sklearn `Pipeline`, so serving can't drift from training.
- **Leakage prevention.** IPs, source ports, flow IDs and timestamps are dropped. Duplicates are
  removed before splitting. SMOTE and selection happen inside training only.
- **Macro-F1 and per-class recall, not accuracy.** Most traffic is benign.
- **Linear SVM (calibrated), subsampled.** A kernel SVM is O(n²) and won't finish on millions
  of flows.
- **Two-part unknown-attack detector.** The Isolation Forest finds unusual *combinations*. The
  range rule catches a single feature more than 10× beyond anything benign, which the forest
  can dilute.
- **Template-based stories, not an LLM.** They're deterministic, fast, testable, and every
  number is real.
- **Quality gate.** `tests/test_model_quality.py` fails if macro-F1 or any attack's recall drops
  below the thresholds in `configs/train.yaml`.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/predict?store=true` | `{"features": {...}}` → label, confidence, story, severity, SHAP, incident |
| POST | `/predict/batch` | `{"flows": [...]}` |
| GET | `/predictions`, `/predictions/{id}` | History (filter `attacks_only`, `incident_id`) |
| GET | `/incidents`, `/incidents/{id}` | Incidents with their alerts |
| POST | `/feedback/{id}` | `{"verdict": "confirmed" \| "false_positive"}` |
| GET / POST | `/feedback/summary`, `/feedback/export` | Analyst precision; labelled CSV for retraining |
| GET | `/drift` | PSI per feature over recent flows |
| GET | `/stats`, `/model`, `/models`, `/baseline`, `/reports`, `/health` | Dashboard data |

See [BUILD.md](BUILD.md) for the full build plan, demo script and report checklist.
