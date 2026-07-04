"""Fake News Prediction System — Streamlit application."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from utils.evaluation import get_prediction_details
from utils.preprocessing import clean_text

PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "raw"

PAGE_NAMES = [
    "Home",
    "Dataset Overview",
    "EDA",
    "Predict",
    "Model Performance",
]

st.set_page_config(
    page_title="Fake News Prediction System",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_model():
    model_path = MODELS_DIR / "best_model.joblib"
    if not model_path.exists():
        return None
    return joblib.load(model_path)


@st.cache_data
def load_json(path_str: str) -> dict:
    path = Path(path_str)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_sample_dataset() -> pd.DataFrame | None:
    fake_path = DATA_DIR / "Fake.csv"
    true_path = DATA_DIR / "True.csv"
    if not fake_path.exists() or not true_path.exists():
        return None

    fake_df = pd.read_csv(fake_path).head(200)
    true_df = pd.read_csv(true_path).head(200)
    fake_df["label"] = "Fake"
    true_df["label"] = "Real"
    return pd.concat([fake_df, true_df], ignore_index=True)


def render_metric_cards(metrics: dict) -> None:
    cols = st.columns(4)
    cards = [
        ("Accuracy", metrics.get("accuracy", 0.0)),
        ("Precision", metrics.get("precision", 0.0)),
        ("Recall", metrics.get("recall", 0.0)),
        ("F1 Score", metrics.get("f1", 0.0)),
    ]
    for column, (label, value) in zip(cols, cards):
        column.metric(label, f"{value:.2%}" if isinstance(value, float) else value)


def render_prediction_result(result: dict) -> None:
    is_fake = result["label"] == "Fake"
    color = "#d9534f" if is_fake else "#5cb85c"
    confidence = (
        f"{result['confidence']:.1%}"
        if result["confidence"] is not None
        else "N/A"
    )

    st.markdown(
        f"""
        <div style="padding: 1.5rem; border-radius: 12px; background: {color}22;
                    border: 1px solid {color};">
            <h3 style="margin: 0; color: {color};">Prediction: {result['label']} News</h3>
            <p style="margin-top: 0.5rem;">Confidence: {confidence}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if result["prob_fake"] is not None:
        prob_df = pd.DataFrame(
            {
                "Class": ["Fake", "Real"],
                "Probability": [result["prob_fake"], result["prob_real"]],
            }
        )
        st.bar_chart(prob_df, x="Class", y="Probability")


def page_home() -> None:
    st.title("Fake News Prediction System")
    st.markdown(
        """
        An end-to-end machine learning application that classifies news articles as
        **Fake** or **Real** using TF-IDF text features and optimized scikit-learn models.

        This capstone project follows the full ML lifecycle: data preparation, EDA,
        model comparison, hyperparameter tuning, and deployment.
        """
    )

    dataset_info = load_json(str(REPORTS_DIR / "dataset_info.json"))
    metadata = load_json(str(MODELS_DIR / "training_metadata.json"))

    if dataset_info:
        cols = st.columns(4)
        cols[0].metric("Total Articles", f"{dataset_info.get('total_rows', 0):,}")
        cols[1].metric("Fake Articles", f"{dataset_info.get('fake_count', 0):,}")
        cols[2].metric("Real Articles", f"{dataset_info.get('real_count', 0):,}")
        cols[3].metric("Best Model", metadata.get("model_name", "Not trained yet"))
    else:
        st.info(
            "Train the model with `python train.py` after placing the dataset in `data/raw/`."
        )

    st.subheader("Problem Statement")
    st.write(
        "Misinformation spreads rapidly online. This system helps analysts and readers "
        "screen article text and receive a prediction with confidence scores."
    )

    st.subheader("Technologies Used")
    st.markdown(
        """
        - Python, Pandas, NumPy
        - Scikit-learn (Logistic Regression, Decision Tree, Random Forest, SVM)
        - TF-IDF feature engineering
        - Joblib model serialization
        - Streamlit deployment
        """
    )


def page_dataset() -> None:
    st.title("Dataset Overview")
    dataset_info = load_json(str(REPORTS_DIR / "dataset_info.json"))
    sample_df = load_sample_dataset()

    st.markdown(
        """
        **Source:** [ISOT Fake News Dataset](https://www.kaggle.com/datasets/atharvaingle/fake-news-classification-dataset)

        The dataset contains two CSV files:
        - `Fake.csv` — articles from unreliable sources
        - `True.csv` — articles from Reuters
        """
    )

    if dataset_info:
        st.subheader("Dataset Summary")
        summary_df = pd.DataFrame(
            {
                "Metric": [
                    "Total rows",
                    "Fake articles",
                    "Real articles",
                    "Dropped empty rows",
                    "Dropped outlier rows",
                    "Average text length",
                    "Average word count",
                ],
                "Value": [
                    dataset_info.get("total_rows"),
                    dataset_info.get("fake_count"),
                    dataset_info.get("real_count"),
                    dataset_info.get("dropped_empty"),
                    dataset_info.get("dropped_outliers"),
                    round(dataset_info.get("avg_text_length", 0), 1),
                    round(dataset_info.get("avg_word_count", 0), 1),
                ],
            }
        )
        st.dataframe(summary_df, width="stretch", hide_index=True)

        dist_path = FIGURES_DIR / "target_distribution.png"
        if dist_path.exists():
            st.image(str(dist_path), caption="Class distribution", width="stretch")

    if sample_df is not None:
        st.subheader("Sample Records")
        st.dataframe(
            sample_df[["title", "subject", "date", "label"]].head(15),
            width="stretch",
        )
    else:
        st.warning("Dataset files not found in `data/raw/`.")


def page_eda() -> None:
    st.title("Exploratory Data Analysis")
    figures = [
        ("target_distribution.png", "Class balance between fake and real articles."),
        ("text_length_histogram.png", "Distribution of article text length by label."),
        ("word_count_boxplot.png", "Word count patterns and outliers by label."),
        ("correlation_heatmap.png", "Correlation between engineered numeric features."),
        ("title_vs_text_scatter.png", "Relationship between title length and body length."),
    ]

    for filename, caption in figures:
        path = FIGURES_DIR / filename
        if path.exists():
            st.image(str(path), caption=caption, width="stretch")
        else:
            st.info(f"Run `python train.py` to generate `{filename}`.")


def page_predict() -> None:
    st.title("Prediction System")
    model = load_model()

    if model is None:
        st.error("Trained model not found. Run `python train.py` first.")
        return

    with st.form("prediction_form"):
        title = st.text_input("Article Title", placeholder="Enter the news headline")
        text = st.text_area(
            "Article Text",
            height=220,
            placeholder="Paste the full article body here",
        )
        submitted = st.form_submit_button("Analyze Article", type="primary")

    if submitted:
        if not title.strip() and not text.strip():
            st.warning("Please provide a title or article text.")
            return

        combined = clean_text(f"{title.strip()} {text.strip()}")
        render_prediction_result(get_prediction_details(model, combined))


def page_metrics() -> None:
    st.title("Model Performance Metrics")
    comparison = load_json(str(REPORTS_DIR / "model_comparison.json"))
    optimized = load_json(str(REPORTS_DIR / "optimized_metrics.json"))

    if comparison:
        st.subheader("Model Comparison")
        table_rows = [
            {
                "Model": model_name,
                "Accuracy": metrics.get("accuracy"),
                "Precision": metrics.get("precision"),
                "Recall": metrics.get("recall"),
                "F1": metrics.get("f1"),
                "ROC AUC": metrics.get("roc_auc"),
            }
            for model_name, metrics in comparison.items()
        ]
        comparison_df = pd.DataFrame(table_rows)
        st.dataframe(
            comparison_df.style.format(
                {
                    "Accuracy": "{:.2%}",
                    "Precision": "{:.2%}",
                    "Recall": "{:.2%}",
                    "F1": "{:.2%}",
                    "ROC AUC": "{:.2%}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

        chart_path = FIGURES_DIR / "model_comparison.png"
        if chart_path.exists():
            st.image(str(chart_path), width="stretch")

    if optimized:
        st.subheader("Optimized Best Model")
        st.write(f"Selected model: **{optimized.get('model_name', 'N/A')}**")
        render_metric_cards(optimized)

        if optimized.get("best_params"):
            st.json(optimized["best_params"])

        for filename in ("optimized_confusion_matrix.png", "optimized_roc_curve.png"):
            path = FIGURES_DIR / filename
            if path.exists():
                st.image(str(path), width="stretch")

    if not comparison and not optimized:
        st.info("No metrics found. Run `python train.py` to generate evaluation reports.")


def main() -> None:
    with st.sidebar:
        st.markdown("## Navigation")
        page_name = st.radio(
            "Go to",
            PAGE_NAMES,
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.caption("DawoodTech Capstone — Week 8")

    page_functions = {
        "Home": page_home,
        "Dataset Overview": page_dataset,
        "EDA": page_eda,
        "Predict": page_predict,
        "Model Performance": page_metrics,
    }
    page_functions[page_name]()


if __name__ == "__main__":
    main()
