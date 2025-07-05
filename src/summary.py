#!/usr/bin/env python3
"""
Build per-cluster R/F/M summary and (optionally) verify dataset size.
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd

LABELS = {
    0: ("Inactive long-tail",  "Low-cost win-back"),
    1: ("VIP / Loyal",         "Loyalty perks, bundles"),
    2: ("Potential loyalists", "Nurture to VIP"),
    3: ("At-Risk",             "Re-engage before churn"),
}

def build_summary(rfm_path: Path) -> pd.DataFrame:
    if not rfm_path.exists():
        sys.exit(f"❌ RFM file not found: {rfm_path}")
    rfm = pd.read_csv(rfm_path, compression="infer")
    tbl = (rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]]
              .mean()
              .round({"Recency": 0, "Frequency": 1, "Monetary": 0})
              .astype({"Recency": int, "Monetary": int}))
    labels = pd.DataFrame.from_dict(LABELS, orient="index",
                                    columns=["Label", "Action"])
    return tbl.join(labels)[["Recency", "Frequency", "Monetary", "Label", "Action"]]

def check_dataset(clean_path: Path) -> None:
    if not clean_path.exists():
        sys.exit(f"❌ Cleaned CSV not found: {clean_path}")
    df = pd.read_csv(clean_path, compression="infer")
    invoices, customers, products = len(df), df["CustomerID"].nunique(), df["StockCode"].nunique()

    print("\nDataset check")
    print("-------------")
    print(f"Invoices (rows)  : {invoices:>8,}")
    print(f"Unique customers : {customers:>8,}")
    print(f"Unique products  : {products:>8,}")

    match = (invoices, customers, products) == (354_321, 3_920, 3_645)
    print("✅ Counts match poster." if match else "⚠️  Counts differ – double-check!")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rfm",   required=True, help="Path to rfm_clusters.csv.gz")
    p.add_argument("--clean", help="Path to online_retail_clean.csv.gz (optional)")
    p.add_argument("--out",   help="CSV output path (optional)")
    args = p.parse_args()

    summary = build_summary(Path(args.rfm))

    # pretty print
    try:
        print(summary.to_markdown(index=True))
    except ImportError:
        print(summary.to_string(index=True))

    # save if requested
    if args.out:
        summary.to_csv(args.out, index=True)
        print(f"\nSaved → {args.out}")

    # dataset verification
    if args.clean:
        check_dataset(Path(args.clean))
