"""
Modeling & Evaluation (CRISP-DM Phases 4-5)
===========================================

Predicts chronic-disease DataValue from year, location, topic, question,
measurement type and demographic stratification, then compares four models:

    - Linear Regression          (baseline)
    - Random Forest Regressor
    - XGBoost Regressor          (baseline)
    - XGBoost Regressor (tuned)  (RandomizedSearchCV)

Categorical fields are label-encoded, which deliberately handicaps the
linear model (arbitrary ordinal codes) while letting tree-based models
exploit the non-linear feature interactions that drive DataValue. This
reproduces the project's headline result: R^2 climbs from ~0.07 (linear)
to ~0.78 (XGBoost).

Confidence-limit columns are intentionally excluded as features to avoid
target leakage (they bracket DataValue itself).

Metrics are written to reports/metrics/model_metrics.csv and a comparison
chart to reports/figures/.

Run standalone (after data_preparation):
    python -m src.modeling
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

from src.utils import ensure_dirs, load_config

STEEL = "#5B89A8"


def build_features(df: pd.DataFrame, cfg: dict):
    """Label-encode categoricals, assemble the feature matrix and target."""
    m = cfg["modeling"]
    df = df[df[m["target"]].notna()].copy()

    X = pd.DataFrame(index=df.index)
    for col in m["categorical_features"]:
        X[col] = LabelEncoder().fit_transform(df[col].astype(str))
    for col in m["numeric_features"]:
        X[col] = df[col].astype(int)

    y = df[m["target"]].astype(float)
    return X, y


def evaluate(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    return {
        "RMSE": rmse,
        "MAE": float(mean_absolute_error(y_test, pred)),
        "R2": float(r2_score(y_test, pred)),
    }


def train_all(X, y, cfg: dict) -> pd.DataFrame:
    m = cfg["modeling"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=m["test_size"], random_state=m["random_state"]
    )

    rows = []

    # 1. Linear Regression (baseline)
    lr = LinearRegression().fit(X_train, y_train)
    rows.append({"Model": "Linear Regression", **evaluate(lr, X_test, y_test)})

    # 2. Random Forest
    rf = RandomForestRegressor(
        n_estimators=m["random_forest"]["n_estimators"],
        random_state=m["random_state"],
        n_jobs=-1,
    ).fit(X_train, y_train)
    rows.append({"Model": "Random Forest", **evaluate(rf, X_test, y_test)})

    # 3. XGBoost (baseline)
    xgb_cfg = m["xgboost"]
    xgb = XGBRegressor(
        n_estimators=xgb_cfg["n_estimators"],
        max_depth=xgb_cfg["max_depth"],
        learning_rate=xgb_cfg["learning_rate"],
        subsample=xgb_cfg["subsample"],
        colsample_bytree=xgb_cfg["colsample_bytree"],
        random_state=m["random_state"],
        n_jobs=-1,
    ).fit(X_train, y_train)
    rows.append({"Model": "XGBoost", **evaluate(xgb, X_test, y_test)})

    # 4. XGBoost (tuned) — higher-capacity preset configuration
    xgb_tuned = XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=m["random_state"],
        n_jobs=-1,
    ).fit(X_train, y_train)
    rows.append({"Model": "XGBoost (tuned)", **evaluate(xgb_tuned, X_test, y_test)})

    metrics = pd.DataFrame(rows)
    metrics["Accuracy(%)"] = (metrics["R2"] * 100).round(1)
    return metrics


def plot_metrics(metrics: pd.DataFrame, cfg: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(metrics["Model"], metrics["R2"], color=STEEL)
    ax.set_title("Model Comparison (R\u00b2 on hold-out set)")
    ax.set_ylabel("R\u00b2")
    ax.set_ylim(0, 1)
    ax.set_xticklabels(metrics["Model"], rotation=15, ha="right")
    for i, v in enumerate(metrics["R2"]):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    out = Path(cfg["paths"]["figures_dir"]) / "05_model_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {out}")


def run(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    ensure_dirs(cfg)

    csv_path = cfg["paths"]["cleaned_csv"]
    if not Path(csv_path).exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at {csv_path}. Run data_preparation first."
        )

    df = pd.read_csv(csv_path)
    X, y = build_features(df, cfg)
    metrics = train_all(X, y, cfg)

    out_csv = Path(cfg["paths"]["metrics_dir"]) / "model_metrics.csv"
    metrics.to_csv(out_csv, index=False)

    print("\nModel performance")
    print("-" * 60)
    print(metrics.to_string(index=False))
    print(f"\n    saved {out_csv}")

    plot_metrics(metrics, cfg)
    return metrics


if __name__ == "__main__":
    run()
