#!/usr/bin/env python3
"""
Cluster R, F, M metrics with K-Means.

CLI
---
python src/clustering.py --input data/rfm_scores.csv.gz --k 4
"""

from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# --------------------------------------------------------------------------- #
def smart_read(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, compression="infer")


def scale_features(rfm: pd.DataFrame) -> np.ndarray:
    scaler = MinMaxScaler()
    return scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])


def elbow_silhouette(X: np.ndarray, k_min: int, k_max: int, out_dir: Path) -> int:
    inertias, sils = [], []
    ks = range(k_min, k_max + 1)
    for k in ks:
        kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42).fit(X)
        inertias.append(kmeans.inertia_)
        sils.append(
            silhouette_score(X, kmeans.labels_) if k > 1 else np.nan
        )  # silhouette undefined for k=1

    # Plot elbow
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(ks, inertias, marker="o")
    plt.title("Elbow Curve")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    sns.despine()
    file = out_dir / "kmeans_elbow.png"
    plt.savefig(file, dpi=150, bbox_inches="tight")
    plt.close()
    print("📐  Saved", file)

    # Print silhouette table
    print("\n🔍  Silhouette scores")
    for k, s in zip(ks, sils):
        print(f"  k={k:<2}  score={s:.3f}")

    # Simple heuristic: choose k with max silhouette in range
    best_k = ks[int(np.nanargmax(sils))]
    print(f"\n⭐  Best k by silhouette ≈ {best_k}")
    return best_k


def run_kmeans(X: np.ndarray, k: int) -> np.ndarray:
    model = KMeans(n_clusters=k, n_init="auto", random_state=42)
    return model.fit_predict(X)


def main(args: argparse.Namespace) -> None:
    in_path  = Path(args.input)
    out_path = Path(args.output).with_suffix(".csv.gz")
    fig_dir  = Path(args.figure_dir)

    rfm = smart_read(in_path)
    X = scale_features(rfm)

    # If k not specified → search 2..10 and pick best
    if args.k:
        k_final = args.k
    else:
        k_final = elbow_silhouette(X, args.k_min, args.k_max, fig_dir)

    # Final model
    labels = run_kmeans(X, k_final)
    rfm["Cluster"] = labels

    # Save
    rfm.to_csv(out_path, index=False, compression="gzip")
    print(f"\n✅ Clustered data saved → {out_path}  (k={k_final})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="K-Means clustering on RFM metrics")
    p.add_argument("--input",  required=True, help="Path to rfm_scores CSV/Parquet")
    p.add_argument("--output", default="data/rfm_clusters", help="Basename for output")
    p.add_argument("--figure-dir", default="reports/figures", help="Folder for elbow PNG")
    p.add_argument("--k", type=int, help="Fix k (skip search)")
    p.add_argument("--k-min", type=int, default=2, help="Min k for search")
    p.add_argument("--k-max", type=int, default=10, help="Max k for search")
    main(p.parse_args())
