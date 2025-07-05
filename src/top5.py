#!/usr/bin/env python3
"""
Return the five customers with the highest order Frequency.

Usage
-----
python src/top5_frequency.py \
    --input  data/rfm_scores.csv.gz \
    --output reports/top5_frequency.csv      # optional
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def top_five(path: Path) -> pd.DataFrame:
    """Load RFM CSV and return the five highest-Frequency rows."""
    df = pd.read_csv(path, compression="infer")
    return (df.sort_values("Monetary", ascending=False)
              .head(5)
              .reset_index(drop=True))


def main(args: argparse.Namespace) -> None:
    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    best = top_five(in_path)

    # ---- console output ----
    print("\n📊  Top 5 customers by order frequency\n")
    print(best[["CustomerID", "Frequency", "Recency", "Monetary"]].to_string(index=False))

    # ---- optional CSV ----
    if args.output:
        out_path = Path(args.output)
        best.to_csv(out_path, index=False)
        print(f"\n✅  Saved → {out_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find top-5 high-frequency customers")
    parser.add_argument("--input", required=True, help="Path to rfm_scores.csv(.gz)")
    parser.add_argument("--output", help="Optional CSV for the result")
    main(parser.parse_args())

13089,  14104, 13506, 13055