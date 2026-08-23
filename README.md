# Predicting U.S. Chronic Disease Trends — CRISP-DM Pipeline

An end-to-end data pipeline that cleans the CDC **Chronic Disease Indicators (CDI)** dataset with **PySpark**, explores it, and models chronic-disease burden with **scikit-learn** and **XGBoost**. Built around the CRISP-DM framework: Business Understanding → Data Understanding → Data Preparation → Modeling → Evaluation.

The project focuses on four leading chronic diseases — **Cancer, Cardiovascular Disease, Diabetes, and Chronic Obstructive Pulmonary Disease (COPD)** — across all U.S. states and territories.

---

## Why this project

Chronic diseases account for roughly 90% of U.S. healthcare spending. The CDI dataset is a large, multi-topic public-health surveillance resource, but in raw form it ships as **309,215 rows × 34 columns**, most of them empty or metadata. This pipeline turns that raw extract into a clean, disease-focused analytical dataset and uses it to forecast disease burden by state, year, and demographic group — the kind of signal a public-health team would use to target interventions.

---

## Architecture

The work is split the way it would be in production: **Spark for scale-out cleaning, Python for modeling on the reduced set.**

```
                  ┌──────────────────────────┐
  raw CDI CSV ──▶ │  data_preparation.py     │  PySpark ETL (7 cleaning steps)
  309,215 × 34    │  (src/)                  │  ─▶ curated Parquet + CSV
                  └────────────┬─────────────┘     69,272 × 13
                               │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
        ┌─────────────────┐        ┌─────────────────────┐
        │  eda.py         │        │  modeling.py        │
        │  matplotlib     │        │  sklearn + XGBoost  │
        │  ─▶ figures     │        │  ─▶ metrics + chart │
        └─────────────────┘        └─────────────────────┘
```

- **PySpark** handles the heavy ETL on the full raw file (multi-line quoted fields, type casting, de-duplication, partitioned Parquet output).
- **pandas / scikit-learn / XGBoost** handle EDA and modeling on the curated 69K-row set.
- `run_pipeline.py` chains all three stages; each stage can also run on its own.

---

## Data preparation (PySpark)

The cleaning pipeline in [`src/data_preparation.py`](src/data_preparation.py) applies seven steps and reproduces the documented shape reduction exactly:

| # | Step | Effect |
|---|------|--------|
| 1 | Drop empty / metadata columns | 34 → 13 columns |
| 2 | Filter to the four target disease topics | 309,215 → 97,336 rows |
| 3 | Drop rows with a missing `DataValue` | 97,336 → 69,272 rows |
| 4 | Remove duplicate surveillance records | 69,272 rows |
| 5 | Standardize dtypes (years → int, values → double) | — |
| 6 | Re-assess nulls (only CI columns remain, ~17,648 each) | — |
| 7 | Persist curated dataset (Parquet partitioned by `Topic` + single CSV) | **69,272 × 13** |

Confidence-interval columns keep their NaNs on purpose — a CI isn't reported for every indicator, so those gaps are expected rather than a quality problem.

---

## Exploratory analysis

Figures land in `reports/figures/`:

- **Average burden by topic** — COPD carries the highest mean `DataValue`, followed by Cardiovascular Disease, Cancer, then Diabetes.
- **Record frequency by topic** — Cardiovascular Disease and Cancer are the most heavily monitored.
- **Reporting over time** — total reported values dip toward 2020–2021, consistent with pandemic-era disruptions to health-service reporting.
- **Records by stratification** — how coverage splits across Sex, Race/Ethnicity, Age, and Overall.

---

## Modeling & evaluation

**Task:** predict `DataValue` (disease burden) from year, location, topic, question, measurement type, and demographic stratification.

Categorical fields are label-encoded, which deliberately handicaps the linear baseline (arbitrary ordinal codes) while letting tree-based models exploit the non-linear interactions that actually drive burden. Confidence-limit columns are **excluded** as features to avoid target leakage.

### Results (original project submission)

| Model | RMSE | MAE | R² |
|-------|-----:|----:|---:|
| Linear Regression | 26,509 | 4,639 | 0.069 |
| Random Forest | 14,704 | 951 | 0.714 |
| **XGBoost** | **12,973** | **1,176** | **0.777** |
| XGBoost (tuned) | 13,923 | 1,082 | 0.743 |

**Headline:** R² rises from ~0.07 (linear) to ~0.78 (XGBoost) — tree-based models capture feature interactions that a linear model cannot.

### Reproducibility note

Running this reconstructed pipeline with `random_state=42` produces results consistent with the original submission and the same LR → tree performance jump:

| Model | R² (this pipeline) |
|-------|-----:|
| Linear Regression | ~0.02 |
| Random Forest | ~0.73 |
| XGBoost | ~0.80 |
| XGBoost (tuned) | ~0.79 |

Exact figures vary slightly with feature encoding and de-duplication order; the story — linear baseline near zero, XGBoost around 0.78–0.80 — is stable. The live numbers from your run are written to `reports/metrics/model_metrics.csv`.

---

## Project structure

```
chronic-disease-indicators/
├── config/
│   └── config.yaml              # paths, Spark settings, feature lists, model params
├── data/
│   ├── raw/                     # place the raw CDI CSV here (git-ignored)
│   └── processed/               # curated Parquet + CSV (generated)
├── src/
│   ├── data_preparation.py      # PySpark ETL — the 7 cleaning steps
│   ├── eda.py                   # exploratory figures
│   ├── modeling.py              # LR / RF / XGBoost baseline + tuned
│   └── utils.py                 # config loading + SparkSession
├── notebooks/
│   └── 01_walkthrough.ipynb     # narrative end-to-end walkthrough
├── reports/
│   ├── figures/                 # generated charts
│   └── metrics/                 # model_metrics.csv
├── run_pipeline.py              # orchestrator
├── requirements.txt
└── README.md
```

---

## Getting started

### 1. Install

```bash
python -m venv .venv && source .venv/bin/activate    # optional
pip install -r requirements.txt
```

PySpark needs a JDK (Java 8/11/17/21). Verify with `java -version`.

### 2. Get the data

Download **U.S. Chronic Disease Indicators** from the CDC and place the CSV at:

```
data/raw/U_S__Chronic_Disease_Indicators.csv
```

Source: https://catalog.data.gov/dataset/u-s-chronic-disease-indicators

### 3. Run

```bash
python run_pipeline.py                # full pipeline: clean → EDA → model
python run_pipeline.py --skip-clean   # reuse curated data, re-run EDA + model
python run_pipeline.py --only clean   # PySpark ETL only
python run_pipeline.py --only model   # modeling only
```

Outputs: curated data in `data/processed/`, figures in `reports/figures/`, metrics in `reports/metrics/`.

---

## Tech stack

**PySpark** · **XGBoost** · **scikit-learn** · **pandas** · **matplotlib** · Parquet · CRISP-DM

## Data source

CDC U.S. Chronic Disease Indicators (CDI) — a public-domain public health
surveillance dataset. Accessed as a filtered extract (309,215 rows x 34 columns).

- Dataset (Data.gov): https://catalog.data.gov/dataset/u-s-chronic-disease-indicators
- About CDI (CDC): https://www.cdc.gov/cdi/index.html

Note: This is a filtered subset of the full CDI release, so row counts may
differ from the complete dataset downloaded directly from the source.
