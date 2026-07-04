"""Data loading, cleaning, and feature engineering for fake news text."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
HTML_PATTERN = re.compile(r"<[^>]+>")
NON_ALPHA_PATTERN = re.compile(r"[^a-zA-Z0-9\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalize raw article text for modeling."""
    if not isinstance(text, str):
        return ""

    cleaned = text.lower()
    cleaned = HTML_PATTERN.sub(" ", cleaned)
    cleaned = URL_PATTERN.sub(" ", cleaned)
    cleaned = NON_ALPHA_PATTERN.sub(" ", cleaned)
    cleaned = WHITESPACE_PATTERN.sub(" ", cleaned).strip()
    return cleaned


class TextPreprocessor:
    """Load, clean, and prepare ISOT fake news data."""

    def __init__(self, data_dir: Path | str | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR

    def load_dataset(self) -> pd.DataFrame:
        """Load and merge Fake.csv and True.csv with binary labels."""
        fake_path = self.data_dir / "Fake.csv"
        true_path = self.data_dir / "True.csv"

        if not fake_path.exists() or not true_path.exists():
            raise FileNotFoundError(
                "Dataset not found. Place Fake.csv and True.csv in data/raw/. "
                "Download from: https://www.kaggle.com/datasets/atharvaingle/fake-news-classification-dataset"
            )

        fake_df = pd.read_csv(fake_path)
        true_df = pd.read_csv(true_path)
        fake_df.columns = fake_df.columns.str.strip()
        true_df.columns = true_df.columns.str.strip()

        fake_df["label"] = 0
        true_df["label"] = 1

        df = pd.concat([fake_df, true_df], ignore_index=True)
        df["title"] = df["title"].fillna("").astype(str)
        df["text"] = df["text"].fillna("").astype(str)
        df["combined_text"] = (
            df["title"].str.strip() + " " + df["text"].str.strip()
        ).str.strip()
        df["combined_text"] = df["combined_text"].map(clean_text)

        before = len(df)
        df = df[df["combined_text"].str.len() > 0].copy()
        dropped_empty = before - len(df)

        df["text_length"] = df["combined_text"].str.len()
        df["title_length"] = df["title"].str.len()
        df["word_count"] = df["combined_text"].str.split().str.len()

        length_cap = int(df["text_length"].quantile(0.99))
        before_cap = len(df)
        df = df[df["text_length"] <= length_cap].copy()
        dropped_outliers = before_cap - len(df)

        df.attrs["dropped_empty"] = dropped_empty
        df.attrs["dropped_outliers"] = dropped_outliers
        df.attrs["length_cap"] = length_cap
        return df.reset_index(drop=True)

    def dataset_summary(self, df: pd.DataFrame) -> dict[str, Any]:
        """Return dataset metadata for reports and the Streamlit app."""
        label_counts = df["label"].value_counts().to_dict()
        return {
            "total_rows": int(len(df)),
            "fake_count": int(label_counts.get(0, 0)),
            "real_count": int(label_counts.get(1, 0)),
            "columns": ["title", "text", "subject", "date", "label"],
            "dropped_empty": int(df.attrs.get("dropped_empty", 0)),
            "dropped_outliers": int(df.attrs.get("dropped_outliers", 0)),
            "length_cap": int(df.attrs.get("length_cap", 0)),
            "avg_text_length": float(df["text_length"].mean()),
            "avg_word_count": float(df["word_count"].mean()),
        }


def load_dataset(data_dir: Path | str | None = None) -> pd.DataFrame:
    """Convenience wrapper around TextPreprocessor.load_dataset."""
    return TextPreprocessor(data_dir=data_dir).load_dataset()


def get_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Split cleaned text and labels into train/test sets."""
    x = df["combined_text"]
    y = df["label"]
    return train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def build_vectorizer() -> TfidfVectorizer:
    """Create the shared TF-IDF vectorizer used by all models."""
    return TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        stop_words="english",
    )


def build_pipeline(estimator: Any) -> Pipeline:
    """Wrap an estimator with the shared TF-IDF vectorizer."""
    return Pipeline(
        [
            ("tfidf", build_vectorizer()),
            ("clf", estimator),
        ]
    )
