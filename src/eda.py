"""
Exploratory Data Analysis (CRISP-DM Phase 2)
============================================

Reproduces the project's EDA figures from the curated dataset:

    - Average DataValue (disease burden) by topic
    - Record frequency by disease topic
    - Chronic disease reporting over time
    - Distribution of records by stratification category

Figures are written to reports/figures/.

Run standalone (after data_preparation):
    python -m src.eda
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import pandas as pd

from src.utils import ensure_dirs, load_config

plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "axes.grid": True})
STEEL = "#5B89A8"


def load_clean(cfg: dict) -> pd.DataFrame:
    csv_path = cfg["paths"]["cleaned_csv"]
    if not Path(csv_path).exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at {csv_path}. Run data_preparation first."
        )
    return pd.read_csv(csv_path)


def _save(fig, cfg: dict, name: str) -> None:
    out = Path(cfg["paths"]["figures_dir"]) / name
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {out}")


def avg_burden_by_topic(df: pd.DataFrame, cfg: dict) -> None:
    s = df.groupby("Topic")["DataValue"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(s.index, s.values, color=STEEL)
    ax.set_title("Average Disease Burden by Topic (cleaned data)")
    ax.set_ylabel("Mean DataValue")
    ax.set_xticklabels(s.index, rotation=20, ha="right")
    _save(fig, cfg, "01_avg_burden_by_topic.png")


def frequency_by_topic(df: pd.DataFrame, cfg: dict) -> None:
    s = df["Topic"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(s.index, s.values, color=STEEL)
    ax.set_title("Record Frequency by Disease Topic")
    ax.set_ylabel("Number of records")
    ax.set_xticklabels(s.index, rotation=20, ha="right")
    _save(fig, cfg, "02_frequency_by_topic.png")


def reporting_over_time(df: pd.DataFrame, cfg: dict) -> None:
    s = df.groupby("YearStart")["DataValue"].sum()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(s.index, s.values, marker="o", color=STEEL)
    ax.set_title("Total Reported DataValue Over Time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Sum of DataValue")
    _save(fig, cfg, "03_reporting_over_time.png")


def records_by_stratification(df: pd.DataFrame, cfg: dict) -> None:
    s = df["StratificationCategory1"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(s.values, labels=s.index, autopct="%1.1f%%", startangle=90,
           colors=plt.cm.Blues([0.4, 0.55, 0.7, 0.85][: len(s)]))
    ax.set_title("Records by Stratification Category")
    _save(fig, cfg, "04_records_by_stratification.png")


def run(cfg: dict | None = None) -> None:
    cfg = cfg or load_config()
    ensure_dirs(cfg)
    df = load_clean(cfg)

    print("EDA summary")
    print("-" * 60)
    burden = df.groupby("Topic")["DataValue"].mean().sort_values(ascending=False)
    for topic, val in burden.items():
        print(f"    mean burden | {topic:<40} {val:12,.2f}")
    print(f"    total rows  : {len(df):,}")
    print(f"    year range  : {df['YearStart'].min()}-{df['YearEnd'].max()}")

    avg_burden_by_topic(df, cfg)
    frequency_by_topic(df, cfg)
    reporting_over_time(df, cfg)
    records_by_stratification(df, cfg)


if __name__ == "__main__":
    run()
