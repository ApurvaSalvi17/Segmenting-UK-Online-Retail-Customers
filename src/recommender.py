#!/usr/bin/env python3
"""
Train a TF-IDF item-similarity model and provide a CLI for quick look-ups.

CLI examples
------------
# build vectors and save artefacts
python src/recommender.py --fit --input data/online_retail_clean.csv.gz

# interactive demo
python src/recommender.py --query 85123A
"""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True, parents=True)
VECT_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
MATRIX_PATH = MODEL_DIR / "item_vectors.npz"
MAP_PATH = MODEL_DIR / "stock_to_idx.pkl"


# --------------------------------------------------------------------------- #
def smart_read(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, compression="infer")


def preprocess_descriptions(df: pd.DataFrame) -> pd.Series:
    return (
        df["Description"]
        .fillna("")
        .str.lower()
        .str.replace(r"[^a-z0-9\s]+", "", regex=True)
    )


def train_tfidf(corpus: list[str]) -> tuple[TfidfVectorizer, any]:
    vect = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
    item_vecs = vect.fit_transform(corpus)
    return vect, item_vecs


def build_and_save(df: pd.DataFrame) -> None:
    # Use the first description per StockCode
    first_desc = (
        df.drop_duplicates("StockCode")
          .sort_values("StockCode")
          .set_index("StockCode")
    )
    corpus = preprocess_descriptions(first_desc).tolist()

    # Train TF-IDF
    vect, item_vecs = train_tfidf(corpus)

    # Stock-to-index map
    stock_to_idx = {sc: i for i, sc in enumerate(first_desc.index)}

    # Serialize
    with open(VECT_PATH, "wb") as f:
        pickle.dump(vect, f)
    with open(MAP_PATH, "wb") as f:
        pickle.dump(stock_to_idx, f)
    # sparse matrix saved with scipy’s built-in .npz
    from scipy import sparse
    sparse.save_npz(MATRIX_PATH, item_vecs)

    print(f"✅  Saved vectorizer → {VECT_PATH}")
    print(f"✅  Saved item matrix → {MATRIX_PATH}")
    print(f"✅  Saved stock map   → {MAP_PATH}")
    print(f"🏁  {item_vecs.shape[0]} unique products vectorised.")


# --------------------------------------------------------------------------- #
def load_artifacts():
    from scipy import sparse
    with open(VECT_PATH, "rb") as f:
        vect = pickle.load(f)
    with open(MAP_PATH, "rb") as f:
        stock_to_idx = pickle.load(f)
    item_vecs = sparse.load_npz(MATRIX_PATH)
    idx_to_stock = {v: k for k, v in stock_to_idx.items()}
    return vect, item_vecs, stock_to_idx, idx_to_stock


def recommend(stock_code: str, top_n: int = 5) -> list[str]:
    vect, item_vecs, stock_to_idx, idx_to_stock = load_artifacts()
    idx = stock_to_idx.get(stock_code)
    if idx is None:
        raise ValueError(f"StockCode {stock_code} not found in training set.")

    sims = cosine_similarity(item_vecs[idx], item_vecs).flatten()
    best = sims.argsort()[-top_n - 1 : -1][::-1]
    return [idx_to_stock[i] for i in best]


# --------------------------------------------------------------------------- #
def main(args: argparse.Namespace) -> None:
    if args.fit:
        df = smart_read(Path(args.input))
        build_and_save(df)

    if args.query:
        print("\n🔎  Recommendations for", args.query)
        try:
            recs = recommend(args.query, args.top_n)
            print("→", ", ".join(recs))
        except ValueError as e:
            print(e)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="TF-IDF product recommender")
    p.add_argument("--input", default="data/online_retail_clean.csv.gz",
                   help="Clean transactions file (CSV/Parquet)")
    p.add_argument("--fit", action="store_true", help="Train & save model")
    p.add_argument("--query", help="StockCode to get recommendations for")
    p.add_argument("--top-n", type=int, default=5, help="# similar items to return")
    main(p.parse_args())
