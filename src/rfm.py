#!/usr/bin/env python3
"""
Compute RFM metrics and quintile scores for the Online Retail dataset.

CLI
---
python src/rfm.py --input data/online_retail_clean.csv.gz
"""

from __future__ import annotations
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
from pandas import Timestamp


# --------------------------------------------------------------------------- #
def smart_read(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    # CSV or CSV.GZ
    return pd.read_csv(path, compression="infer", parse_dates=["InvoiceDate"])


def compute_rfm(df: pd.DataFrame, snapshot: Timestamp) -> pd.DataFrame:
    """Return a DF with Recency, Frequency, Monetary and quintile scores."""
    rfm = (
        df.groupby("CustomerID")
          .agg({
              "InvoiceDate": lambda x: (snapshot - x.max()).days,
              "InvoiceNo":   "nunique",
              "TotalPrice":  "sum",
          })
          .rename(columns={
              "InvoiceDate": "Recency",
              "InvoiceNo":   "Frequency",
              "TotalPrice":  "Monetary",
          })
    )

    # --- Quintile scores (1 = bad, 5 = best) ------------------------------ #
    rfm["R"] = pd.qcut(rfm["Recency"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["F"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5,
                       labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M"] = pd.qcut(rfm["Monetary"], 5, labels=[1, 2, 3, 4, 5]).astype(int)

    rfm["RFM_Score"] = rfm[["R", "F", "M"]].sum(axis=1)
    return rfm.reset_index()


def write_outputs(rfm: pd.DataFrame, base: Path) -> None:
    """Save to Parquet if possible else gzip-CSV."""
    try:
        p_path = base.with_suffix(".parquet")
        rfm.to_parquet(p_path, index=False)
        print("✅ Parquet saved →", p_path)
    except Exception:
        c_path = base.with_suffix(".csv.gz")
        rfm.to_csv(c_path, index=False, compression="gzip")
        print("✅ CSV.gz  saved →", c_path)


# --------------------------------------------------------------------------- #
def main(args: argparse.Namespace) -> None:
    in_path  = Path(args.input)
    out_base = Path(args.output).with_suffix("")   # extension added later

    df = smart_read(in_path)

    # choose snapshot
    snapshot = (pd.to_datetime(args.snapshot)
                if args.snapshot
                else df["InvoiceDate"].max() + pd.Timedelta(days=1))

    rfm = compute_rfm(df, snapshot)

    # quick sanity check
    print("\n🔢  RFM summary")
    print(rfm[["Recency", "Frequency", "Monetary"]].describe().round(1))
    print("\n🎲  Quintile counts")
    print(rfm[["R", "F", "M"]].apply(pd.Series.value_counts).fillna(0).astype(int))

    write_outputs(rfm, out_base)
    print("\n🏁  Done —", len(rfm), "customers scored.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compute RFM metrics & scores")
    p.add_argument("--input", required=True, help="Clean CSV/Parquet file")
    p.add_argument("--output", default="data/rfm_scores", help="Basename for output")
    p.add_argument("--snapshot", help="YYYY-MM-DD date to measure Recency from")
    main(p.parse_args())
