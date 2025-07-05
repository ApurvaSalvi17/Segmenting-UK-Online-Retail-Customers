#!/usr/bin/env python3
"""
Clean the UCI Online Retail (Excel) file.
⇢ Prefer Parquet for speed, but gracefully fall back to compressed CSV.

Example
-------
python src/data_prep.py \
    --input  "data/Online Retail 2.xlsx"
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


def load_and_clean(xl_path: Path) -> pd.DataFrame:
    """Load Excel → tidy UK-only transactions with helpful extra columns."""
    print(f"📂 Reading  → {xl_path}")
    df = pd.read_excel(xl_path)

    # 1️⃣ keep only rows with a CustomerID
    df = df.dropna(subset=["CustomerID"])

    # 2️⃣ drop cancelled invoices (InvoiceNo starts with “C”)
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C", na=False)]

    # 3️⃣ focus on United Kingdom
    df = df[df["Country"] == "United Kingdom"]

    # 4️⃣ convert dates and compute revenue
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalPrice"]  = df["Quantity"] * df["UnitPrice"]

    # 5️⃣ sanity filter: positive quantity and price
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    return df


def write_file(df: pd.DataFrame, base: Path) -> Path:
    """
    Attempt Parquet first; if that fails (missing pyarrow/fastparquet)
    fall back to gzip-compressed CSV.
    """
    try:
        parquet_path = base.with_suffix(".parquet")
        df.to_parquet(parquet_path, index=False)
        print(f"✅ Parquet saved  → {parquet_path}")
        return parquet_path
    except Exception as err:
        csv_path = base.with_suffix(".csv.gz")
        df.to_csv(csv_path, index=False, compression="gzip")
        print(
            "⚠️  Parquet failed; wrote compressed CSV instead →", csv_path,
            "\n   Error was:", err.__class__.__name__, "-", err
        )
        return csv_path


def main(args: argparse.Namespace) -> None:
    in_path  = Path(args.input)
    out_base = Path(args.output).with_suffix("")  # strip any extension

    if not in_path.exists():
        sys.exit(f"❌ Input file not found: {in_path}")

    clean_df = load_and_clean(in_path)
    out_path = write_file(clean_df, out_base)

    print(
        f"   Rows:    {clean_df.shape[0]:,}\n"
        f"   Columns: {clean_df.shape[1]}\n"
        f"👍 You can load it later with "
        f"pd.read_parquet('{out_path}') "
        f"or pd.read_csv('{out_path}', compression='gzip')."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean Online Retail dataset")
    parser.add_argument(
        "--input",
        required=True,
        help='Path to Excel file (e.g. "data/Online Retail 2.xlsx")',
    )
    parser.add_argument(
        "--output",
        default="data/online_retail_clean",   # extension added automatically
        help='Output basename; ".parquet" or ".csv.gz" will be appended',
    )
    main(parser.parse_args())
