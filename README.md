# rare-event-prediction-logistic-regression

**Overview**
- This project builds a rare-event prediction model for escalation interactions using logistic regression in Databricks. The workflow emphasizes precision-first outreach selection by applying a threshold-based cutoff that meets a defined precision floor. Includes feature engineering, encoding, and audit artifacts

**Key objectives**:

- Predict escalation-positive interactions from historical call/service data.
- Optimize outreach decisions using precision floor and max recall strategy.
- Provide auditable artifacts for transparency and reproducibility.


**Features**

Data Cleaning & Normalization
- Consolidates PYDESCRIPTION categories based on business rules.

Feature Engineering
- Interaction-level aggregation, recency metrics, rolling call counts.

Encoding
- Multi-hot flags for cleaned PYDESCRIPTION values.
- Target encoding for categorical features.

Model Training
- Logistic Regression with L2 regularization.
- Grid search over C values.

Threshold Selection
- Chooses cutoff for max recall subject to precision floor.

Outputs
- Outreach file for operational use.
- Optional diagnostics: PR curve, Top-K ranking, run log.

**How It Works**

1. Load Data
- Reads interaction-level service data from Databricks volumes.
2. Clean & Normalize
- Applies canonical mapping to PYDESCRIPTION values.
3. Aggregate & Engineer Features
- Builds interaction-level table with recency and frequency metrics.
4. Encode & Split
- Downsamples negatives, splits by time, applies multi-hot and target encoding.
5. Train Model
- Logistic Regression with class weights and feature scaling.
6. Select Threshold
- Finds cutoff that satisfies precision floor and maximizes recall.
7. Export Outputs
- Saves outreach file and optional diagnostics.


**Key Parameters**

- TARGET_PRECISION — Precision floor for outreach (default: 0.40).
- THRESHOLD_CAP — Optional cap on outreach volume.
- SPLIT_BOUNDARY — Date for train/test split.
- TARGET_NEG_POS_MEMBER_RATIO — Downsampling ratio for negatives.


**Outputs Explained**

- outreach_threshold_precision40_nocap.csv - Final outreach list based on threshold selection.
- pydescription_normalization_audit.csv - Before/after mapping for PYDESCRIPTION cleaning.
- pydesc_to_safe_col_map.csv - Maps original descriptions to sanitized column names.
- pr_curve_points.csv - Precision–Recall curve points for diagnostics.
- outreach_topK_500.csv (optional) - Ranked list of top-scoring interactions for review.
- outreach_runs_log.csv (optional) - Run summary for audit and drift monitoring.


**How to Run**

1. Import the notebook into Databricks.
2. Attach to a cluster with Spark + Python environment.
3. Update paths for input data and output directory.
4. Run cells sequentially or use Run All.
5. Check outputs in your Workspace folder or configured path.


**Next Steps**

1. Add MLflow integration for experiment tracking.
2.Extend modeling to include tree-based methods for comparison.
3. Automate daily scoring and outreach file generation.
