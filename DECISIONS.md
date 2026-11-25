# Decision log

Each entry records what was decided, the alternatives, and why. Add an entry whenever a
choice would be hard to explain in a viva six weeks later.

---

### D1. Flow features, not packets
**Decision:** classify CICFlowMeter flow statistics (CIC-IDS2017 MachineLearningCSV).
**Alternatives:** raw packets/PCAP with deep learning; NetFlow v5 only.
**Why:** flow statistics are the standard representation for this benchmark, work on
encrypted traffic, and keep models fast and explainable. Cost: detection only happens once a
flow ends.

### D2. Drop identifiers (IPs, source port, flow ID, timestamp)
**Why:** in CIC-IDS2017 attacks come from a few known hosts at known times. Keeping these
lets a model memorise *who/when* instead of learning *how*, which inflates scores and fails
on any other network. `Destination Port` is **kept for now** but is a known shortcut
(see D9).

### D3. Remove duplicates before splitting
**Why:** CIC-IDS2017 contains many identical rows. If copies land in both train and test the
test score measures memory, not generalisation.

### D4. 70 / 15 / 15 train / validation / test; choose on validation
**Alternatives:** 80/20 train/test (FlowSight's first version picked the best model by test
score).
**Why:** choosing a model by its test score quietly tunes on the test set. Now every choice
(model, k, imbalance strategy) uses validation or cross-validation on training data, and
the test set is used once, for the reported numbers.

### D5. Five classes + Unknown-Anomaly
**Decision:** Benign, DDoS, DoS, PortScan, BruteForce; rare CIC labels (Bot, Web Attack,
Infiltration, Heartbleed) are dropped by default (`drop_other`).
**Why:** DoS and DDoS behave differently and both matter; the rare classes have too few
flows for honest per-class metrics. Anything unlike benign traffic that the classifier
calls Benign becomes `Unknown-Anomaly` (D7).

### D6. Macro-F1 and per-class recall over accuracy
**Why:** ~80% of flows are benign; a model that never alerts scores ~80% accuracy. Macro-F1
weights every class equally, and the CI quality gate also requires a minimum recall for
every attack class.

### D7. Two-part unknown-attack detector
**Decision:** Isolation Forest on benign training flows **plus** a range rule (any feature
>10× beyond anything benign).
**Why:** the forest catches unusual *combinations* but averages over all features, so a single
wildly extreme feature can be diluted; the range rule catches that and is easy to explain.
Threshold = 0.5% of benign training flows (configurable).

### D8. Template-based alert stories, not an LLM
**Why:** deterministic, instant, testable, and every number in the sentence is the real
flow value compared with real benign percentiles. An LLM could phrase it more nicely
but might invent numbers.

### D9. Keep Destination Port (for now)
**Why:** it genuinely helps for brute force (22/21). **Risk:** the model may learn "port 22 =
attack". The EDA notebook and global SHAP report check how much each class depends on it;
train once with and once without it and record the outcome here.

### D10. Linear SVM (calibrated), class-weighted, subsampled
**Alternatives:** RBF-kernel SVC.
**Why:** kernel SVMs are O(n²) and don't finish on millions of flows; without class
weighting they collapse on minority classes. LinearSVC + `class_weight="balanced"` +
calibration for probabilities is the fair version of "SVM" for this data size.

### D11. Severity = base(type) × confidence × burst
**Why:** gives analysts a priority order. Weights live in `configs/severity.yaml` so they
can be argued about and changed without touching code.

### D12. SQLite + FastAPI + Streamlit
**Why:** zero-setup for a student project, one process each, easy to demo. Swap SQLite for
Postgres if multiple API workers are needed.

### D14. Softer class weights for XGBoost (balanced ** 0.5)
**Evidence (CIC-IDS2017 validation split):** with sklearn's "balanced" weights each of the
rare BruteForce flows counted ~50× a benign flow; XGBoost then over-predicted BruteForce
(recall 0.98, **precision 0.29**, macro-F1 0.88). Square-rooted weights: BruteForce F1 0.97+.
Configurable as `xgb_weight_power`.

### D15. Tree-based feature selection (ExtraTrees top-k) instead of ANOVA F
**Evidence:** ANOVA F is linear and univariate; on CIC-IDS2017 it chose 20 near-duplicate
packet-length features and **dropped Destination Port**. Switching to ExtraTrees importance
raised validation macro-F1 from 0.990 to 0.998 (XGBoost, √ weights). The selector stores
its mask at fit time (`TreeTopK`); `SelectFromModel` recomputed importances on every
prediction and cost ~35 ms per flow.

### D16. Cap training rows at 600k (validation/test full size)
**Why:** the full 70% training split is ~1.76M flows; on a 16 GB laptop that risks running
out of memory and produces a ~GB Random Forest. A stratified 600k sample loses almost
nothing (rare classes keep their proportion), while validation and test stay at ~377k each
so reported numbers are not affected by the cap.

### D17. Unknown-attack rule = anomaly detector OR low classifier confidence
**Evidence:** on CIC-IDS2017 the benign-only Isolation Forest flags <0.05% of attack flows;
most attacks look statistically normal flow-by-flow. The classifier's own doubt is a
better signal: "Benign, but < 90% sure" recovers 2/3 of the attacks the classifier missed
on the test set at ~0.5% benign false alarms. Both parts are kept and reported separately;
`reports/zero_day.md` measures the rule on attacks the model never saw.

### D13. Synthetic sample data for development only
**Why:** lets the whole system run and be tested in CI without a 1 GB download. It is
deleted automatically by `scripts/fetch_dataset.py`, and the model card and dataset audit
warn loudly if it's present. Its scores are never reported.
