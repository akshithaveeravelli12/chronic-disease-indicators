"""
Data Preparation (CRISP-DM Phase 3) — PySpark ETL
==================================================

Transforms the raw CDC Chronic Disease Indicators extract
(309,215 rows x 34 cols) into a clean, disease-focused analytical
dataset (69,272 rows x 13 cols).

The seven cleaning steps below mirror the project's CRISP-DM
data-preparation documentation exactly:

    1. Drop irrelevant / empty columns (34 -> 13)
    2. Filter to the four target disease topics
    3. Drop rows with a missing DataValue (the primary measure)
    4. Remove duplicate surveillance records
    5. Standardize data types (years -> int, values -> double)
    6. Re-assess missing values (only CI columns should remain)
    7. Persist the curated dataset (Parquet + single CSV)

Run standalone:
    python -m src.data_preparation
"""
from __future__ import annotations

import shutil
from pathlib import Path

from pyspark.sql import DataFrame, functions as F
from pyspark.sql.types import DoubleType, IntegerType

from src.utils import ensure_dirs, get_spark, load_config


def load_raw(spark, raw_csv: str) -> DataFrame:
    """Read the raw CDI CSV. multiLine + escape handle the quoted, comma-rich
    Question / Stratification text fields safely."""
    return spark.read.csv(
        raw_csv,
        header=True,
        inferSchema=True,
        multiLine=True,
        escape='"',
    )


def clean(df: DataFrame, cfg: dict) -> DataFrame:
    """Apply the seven-step cleaning pipeline and return the curated frame."""
    keep = cfg["keep_columns"]
    topics = cfg["target_topics"]

    start_rows, start_cols = df.count(), len(df.columns)
    print(f"[1] raw shape                 : {start_rows:,} rows x {start_cols} cols")

    # Step 1 — keep only the analytical columns (drops empty/metadata fields).
    df = df.select(*keep)
    print(f"[1] after column selection    : {df.count():,} rows x {len(df.columns)} cols")

    # Step 2 — filter to the four target disease topics.
    df = df.filter(F.col("Topic").isin(topics))
    print(f"[2] after topic filter        : {df.count():,} rows")

    # Step 3 — drop rows with no DataValue (unusable for analysis/modeling).
    df = df.filter(F.col("DataValue").isNotNull())
    print(f"[3] after dropping null value : {df.count():,} rows")

    # Step 4 — remove duplicate surveillance records.
    df = df.dropDuplicates()
    print(f"[4] after de-duplication      : {df.count():,} rows")

    # Step 5 — standardize data types.
    df = (
        df.withColumn("YearStart", F.col("YearStart").cast(IntegerType()))
        .withColumn("YearEnd", F.col("YearEnd").cast(IntegerType()))
        .withColumn("DataValue", F.col("DataValue").cast(DoubleType()))
        .withColumn("LowConfidenceLimit", F.col("LowConfidenceLimit").cast(DoubleType()))
        .withColumn("HighConfidenceLimit", F.col("HighConfidenceLimit").cast(DoubleType()))
    )

    # Step 6 — re-assess missing values (expect only the CI columns).
    print("[6] remaining nulls by column :")
    _report_nulls(df)

    end_rows, end_cols = df.count(), len(df.columns)
    print(f"[7] final shape               : {end_rows:,} rows x {end_cols} cols")
    return df


def _report_nulls(df: DataFrame) -> None:
    total = df.count()
    null_counts = df.select(
        [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]
    ).collect()[0].asDict()
    for col, n in null_counts.items():
        if n > 0:
            pct = 100 * n / total
            print(f"      - {col:<22} {n:>8,} ({pct:4.1f}%)")


def persist(df: DataFrame, cfg: dict) -> None:
    """Write curated data as partitioned Parquet and one consolidated CSV."""
    parquet_path = cfg["paths"]["cleaned_parquet"]
    csv_path = cfg["paths"]["cleaned_csv"]

    # Cache so both writes reuse the same computed frame.
    df = df.cache()

    (
        df.write.mode("overwrite")
        .partitionBy("Topic")
        .parquet(parquet_path)
    )
    print(f"    wrote Parquet -> {parquet_path}")

    # Coalesce to a single CSV for downstream pandas/BI consumption.
    tmp_dir = Path(csv_path + "_tmp")
    (
        df.coalesce(1)
        .write.mode("overwrite")
        .option("header", True)
        .csv(str(tmp_dir))
    )
    part = next(tmp_dir.glob("part-*.csv"))
    shutil.move(str(part), csv_path)
    shutil.rmtree(tmp_dir)
    print(f"    wrote CSV     -> {csv_path}")


def run(cfg: dict | None = None) -> str:
    """Entry point: clean the raw extract and persist the curated dataset.
    Returns the path to the cleaned CSV."""
    cfg = cfg or load_config()
    ensure_dirs(cfg)

    raw_csv = cfg["paths"]["raw_csv"]
    if not Path(raw_csv).exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {raw_csv}. "
            "Download 'U.S. Chronic Disease Indicators' from data.cdc.gov "
            "and place it there (see README)."
        )

    spark = get_spark(cfg)
    try:
        raw = load_raw(spark, raw_csv)
        cleaned = clean(raw, cfg)
        persist(cleaned, cfg)
    finally:
        spark.stop()

    return cfg["paths"]["cleaned_csv"]


if __name__ == "__main__":
    run()
