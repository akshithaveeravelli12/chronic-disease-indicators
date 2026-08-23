"""
End-to-end CRISP-DM pipeline runner
===================================

Chains the three stages:

    1. data_preparation  (PySpark ETL)  -> curated dataset
    2. eda               (matplotlib)   -> figures
    3. modeling          (sklearn/xgb)  -> metrics + comparison chart

Usage:
    python run_pipeline.py                # run everything
    python run_pipeline.py --skip-clean   # reuse existing curated data
    python run_pipeline.py --only model   # run a single stage
"""
from __future__ import annotations

import argparse
import time

from src import data_preparation, eda, modeling
from src.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CRISP-DM pipeline.")
    parser.add_argument("--skip-clean", action="store_true",
                        help="reuse existing curated dataset (skip PySpark ETL)")
    parser.add_argument("--only", choices=["clean", "eda", "model"],
                        help="run only one stage")
    args = parser.parse_args()

    cfg = load_config()

    def banner(text: str) -> None:
        print("\n" + "=" * 70)
        print(f"  {text}")
        print("=" * 70)

    if args.only == "clean":
        banner("STAGE 1/3 — Data preparation (PySpark)")
        data_preparation.run(cfg)
        return
    if args.only == "eda":
        banner("STAGE 2/3 — Exploratory data analysis")
        eda.run(cfg)
        return
    if args.only == "model":
        banner("STAGE 3/3 — Modeling & evaluation")
        modeling.run(cfg)
        return

    t0 = time.time()

    if not args.skip_clean:
        banner("STAGE 1/3 — Data preparation (PySpark)")
        data_preparation.run(cfg)
    else:
        print("Skipping PySpark ETL; reusing existing curated dataset.")

    banner("STAGE 2/3 — Exploratory data analysis")
    eda.run(cfg)

    banner("STAGE 3/3 — Modeling & evaluation")
    modeling.run(cfg)

    banner(f"Pipeline complete in {time.time() - t0:0.1f}s")


if __name__ == "__main__":
    main()
