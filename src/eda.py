#!/usr/bin/env python3
"""
Generate basic EDA figures for the Online Retail dataset.

CLI usage
---------
python src/eda.py --input data/online_retail_clean.csv.gz
"""

from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------------------------------------------------- #
def smart_read(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)

    # CSV / CSV.GZ branch  🔽  add parse_dates
    elif path.suffix in {".csv", ".gz", ".csv.gz"}:
        return pd.read_csv(
            path,
            compression="infer",
            parse_dates=["InvoiceDate"]   # ← add this line
        )

    else:
        raise ValueError(f"Unsupported extension: {path.suffix}")



# ------------------------------ Plot helpers -------------------------------- #
def plot_monthly_orders(df: pd.DataFrame, out_dir: Path) -> None:
    monthly = df.resample("M", on="InvoiceDate")["InvoiceNo"].nunique()
    plt.figure(figsize=(10, 4))
    monthly.plot()
    plt.title("Monthly Unique Invoices")
    plt.ylabel("# Orders")
    plt.xlabel("")
    sns.despine()
    file = out_dir / "monthly_orders.png"
    plt.savefig(file, dpi=150, bbox_inches="tight")
    plt.close()
    print("📈  Saved", file)


def plot_revenue_pareto(df: pd.DataFrame, out_dir: Path) -> None:
    revenue = df.groupby("CustomerID")["TotalPrice"].sum().sort_values(ascending=False)
    cum_pct = revenue.cumsum() / revenue.sum() * 100
    plt.figure(figsize=(8, 5))
    cum_pct.reset_index(drop=True).plot()
    plt.axhline(80, ls="--", color="gray")
    plt.text(len(cum_pct) * 0.78, 82, "80 % line")
    plt.title("Pareto Chart – Cumulative Revenue vs. Customers")
    plt.ylabel("Cumulative % of Revenue")
    plt.xlabel("Customers (sorted)")
    sns.despine()
    file = out_dir / "revenue_pareto.png"
    plt.savefig(file, dpi=150, bbox_inches="tight")
    plt.close()
    print("📊  Saved", file)


def top_products_bar(df: pd.DataFrame, out_dir: Path, n: int = 10) -> None:
    top = (
        df.groupby("Description")["TotalPrice"]
        .sum()
        .sort_values(ascending=False)
        .head(n)
        .sort_values()
    )
    plt.figure(figsize=(8, 5))
    top.plot(kind="barh")
    plt.title(f"Top {n} Products by Revenue")
    plt.xlabel("Revenue (£)")
    sns.despine()
    file = out_dir / "top_products.png"
    plt.savefig(file, dpi=150, bbox_inches="tight")
    plt.close()
    print("🛍️  Saved", file)


# ----------------------------- CLI wrapper ---------------------------------- #
def main(args: argparse.Namespace) -> None:
    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = smart_read(in_path)

    # quick peek
    print(f"🔍  Data shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print("🗓   Date range :", df['InvoiceDate'].min().date(), "→", df['InvoiceDate'].max().date())

    # plots
    plot_monthly_orders(df, out_dir)
    plot_revenue_pareto(df, out_dir)
    top_products_bar(df, out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Create basic EDA plots")
    p.add_argument("--input", required=True, help="Path to clean CSV or Parquet")
    p.add_argument("--output-dir", default="reports/figures", help="Folder for PNGs")
    main(p.parse_args())
