# Rare Event Prediction Logistic Regression

**Author:** Melanie Kim <BR>
**Overview**
- This project builds a rare-event prediction model for escalation interactions using logistic regression in Databricks. The workflow emphasizes precision-first outreach selection by applying a threshold-based cutoff that meets a defined precision floor. The model builds a complete, production‑style pipeline on Spark + scikit‑learn for **rare‑event classification**. 

**Key objectives**:
- Predict escalation-positive interactions from historical call/service data.
- Optimize outreach decisions using precision floor and max recall strategy.
- Provide auditable artifacts for transparency and reproducibility.

**Features**
- Aggregates service‑item level rows into interaction‑level examples.
- Cleans and canonicalizes text reasons (`PYDESCRIPTION`) for robust features.
- Engineers **sequence‑aware, past‑only** features (rolling windows, prior ELE, recency).
- Downsamples negatives at the **member** level and splits data **over time**.
- Encodes categories, adds **multi‑hot flags**, and **target encodes** descriptions.
- Trains **L2 Logistic Regression**, chooses the operating point by **max recall** subject to a **precision floor**, and (optionally) applies a capacity cap.
- Produces **diagnostics** (PR curve, feature importance, grouped permutation, ablations, monthly stability).
- Exports ranked outreach selections and a **run log** of metrics + artifact paths.

**Data & Labeling**
- **Input format:** Parquet, loaded from a configurable path (Databricks Volumes path by default).
- **Label:** `label = int(ELE)` on service‑item rows; interaction label = `max(ELE)` across its service items.
- **Timestamp:** `PXCREATEDATETIME` normalized to `to_timestamp`.
- **Quick context:** Plots **class prevalence** (positive/negative counts).

**Interaction Aggregation**
Builds a per‑interaction table:
- `CALL_TIME = min(PXCREATEDATETIME)` per `(MEMBERID, INTERACTIONID)`.
- `interaction_label = max(ELE)` (int).
- Summaries:
  - `svc_item_cnt` — distinct service items per interaction.
  - `distinct_PYDESCRIPTION_cnt` — distinct cleaned descriptions.
  - `PYDESCRIPTION_primary` — first cleaned description.
  - `PRDCT_CD_set_size` — distinct product codes.
  - `PYDESC_SET` — collected set of cleaned descriptions.


## Feature Engineering (Past‑Only)
Sequence‑aware features computed per interaction in **epoch seconds** (`ts_long`) so time windows are exact:
- **Rolling call counts**: `calls_5d`, `7d`, `14d`, `30d`, `60d`, `90d`.
- **Prior ELE counts** (exclusive of current row): `ele_inter_30d`, `45d`, `60d`, `90d`, `100d`, `110d`, `120d`, `130d`.
- **Recency features**:
  - `days_since_prev_call` (lag of `CALL_TIME` per member).
  - `days_since_ele` (since most recent prior ELE).
  - `ele_recency_exp = exp(-days_since_ele / 30)`.
- **Final NA fills** across engineered columns (defensive zero fills).

## Split & Encoding
### Downsampling
- Negative downsampling at the **member** level using `TARGET_NEG_POS_MEMBER_RATIO` to control size and imbalance.
- Keeps **all positive members**; samples negatives uniformly.

### Time‑based split (prospective)
- **Training pool** (before `SPLIT_BOUNDARY`), then:
  - **train_fit** — model fitting and encoding learning.
  - **validation** — threshold tuning (+ optional calibration).
- **Test** — post‑boundary holdout for final reporting.

### Multi‑hot flags (train vocabulary)
- Build **train‑only** vocabulary from `PYDESC_SET` and create **deterministic**, sanitized flag columns named via `sanitize_col_name`:
  - Replace non‑alphanumeric with `_`, collapse multiple `_`, strip edges.
- Export mapping `pydesc_to_safe_col_map.csv` and **Top‑25 flag frequencies**.

### Target Encoding (TE)
- Train‑only TE for cleaned `PYDESCRIPTION` categories.
- Uses **m‑estimate smoothing** with global prior to stabilize rare labels (average over a prior with weight `m`).
- Aggregated TE features per interaction:
  - `PYDESC_TE_mean`, `PYDESC_TE_max`.

## Indexing & Conversion
- Build **frequency‑ordered** category maps for categorical columns (e.g., `PYDESCRIPTION_primary_idx`, `REGIONCODE_idx`, …).
- Apply maps to train/val/test (left join; `0` reserved for “other”).
- Compose `feature_cols = [*_idx] + numeric base + engineered + multi‑hot + TE`.
- Convert Spark → Pandas with **defensive casting** (`to_numeric`, `fillna`, dtype enforcement).
- **Class weights**:
  - Compute explicit **sample weights** post‑downsampling, or use `class_weight='balanced'` (toggle).

## Modeling & Threshold Selection
- Model: **L2 Logistic Regression** (`lbfgs`, `max_iter=1000`) with **grid** over `C` and a **reproducible seed**.
- **Scaling policy**:
  - `ColumnTransformer`: **standardize continuous features only**; **passthrough** binary flags (multi‑hot and other indicators).
- **Calibration (optional)**:
  - Isotonic calibration on the **validation** set; report **Brier score**.
- **Operating point**:
  - On **validation**, scan `precision_recall_curve` and choose the threshold that meets the **precision floor** and **maximizes recall**.
  - Evaluate precision/recall/AP on **test** at that threshold.
- **Capacity cap**:
  - Apply cap (if set) **after ranking by score**; export both **uncapped** and **capped** selections for transparency.

**Grid search chart**: test recall/precision/AP across `C`, with validation operating points overlaid.

## Diagnostics & Reporting
- **Precision–Recall curve** with **Average Precision (AP)**.
- **Anchors** (e.g., recall ≈ 0.10, 0.20, …, 0.70) for drift monitoring and business communication.
- Export `pr_curve_points.csv` for plotting in BI tools.

## Feature Importance & Stability
- **Standardized coefficients** (comparable due to continuous scaling).
- **Permutation importance**:
  - AP drop, and precision drop at the **operating cutoff**.
- **Grouped permutation**:
  - Groups: description flags, time windows, prior ELE, TE, recency.
- **Ablations**:
  - Remove feature families; re‑train + re‑select threshold on **validation**; evaluate on **test** for operational impact.
- **Monthly stability** (test set):
  - Compute AP‑based permutation drops per **month**; summarize mean/std to track feature consistency over time.

## Outreach Export
Exports ranked outreach selections:

- **Uncapped**: `outreach_threshold_precision{XX}_nocap.csv`.
- **Capped** (if `THRESHOLD_CAP` set): `outreach_threshold_precision{XX}_cap{CAP}.csv`.
- Columns include:
  - `MEMBERID`, `INTERACTIONID`, `score`, `interaction_label`, `rank`,
  - cumulative stats: `cum_tp`, `cum_fp`, `cum_precision`, `cum_recall`.


## Docs & Run Log
If `ENABLE_DOCS=True`:
- **Top‑K** (diagnostics/reporting only): `outreach_topK_{K}.csv`.
- **Run log** (`outreach_runs_log.csv`):
  - timestamp, strategy, target precision, capacity cap, chosen cutoff,
  - AP, precision/recall at cutoff, selected count, TP/FP/FN,
  - artifact file paths (topK, threshold files).
- Lists/prints discovered importance artifacts (e.g., coefficients, permutation, grouped importance, ablations, stability).

## ⚙️ Configuration
All runtime knobs are in a **CONFIG** section (top of the file). Key parameters:
- **Paths**
  - `VOLUME_PATH`: parquet source (can be overridden via `ELE_DATA_PATH` env var).
- **Docs/Exports**
  - `ENABLE_DOCS`: whether to write Top‑K and run log.
  - `K_FOR_EXPORT`: Top‑K size.
- **Splits**
  - `SPLIT_BOUNDARY`: time boundary for **test**.
  - `VAL_BOUNDARY`: time boundary inside training for **validation** (threshold & calibration).
- **Model**
  - `USE_SCALER_CONT_ONLY`: scale continuous features only.
  - `C_GRID`: LR regularization grid.
  - `TARGET_PRECISION`: precision floor (e.g., `0.40`).
  - `THRESHOLD_CAP`: optional outreach capacity cap (int) or `None`.
  - `CALIBRATE_PROBS`: apply isotonic calibration on validation.
- **Imbalance**
  - `USE_EXPLICIT_WEIGHTS`: use sample weights; else `class_weight='balanced'`.
  - `TARGET_NEG_POS_MEMBER_RATIO`: negative downsampling ratio at **member** level.
- **Misc**
  - `SEED`: reproducibility.
  - `PLOT_DPI`: image resolution.

**Next Steps**
1. Add MLflow integration for experiment tracking
2. Extend modeling to include tree-based methods for comparison
3. Automate daily scoring and outreach file generation
