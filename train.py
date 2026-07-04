"""Train, evaluate, optimize, and export fake news classification models."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils.evaluation import (
    compute_metrics,
    cross_validate_model,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_roc_curve,
    save_metrics,
)
from utils.models import MODEL_DISPLAY_NAMES, get_model_builders, optimize_model
from utils.preprocessing import TextPreprocessor, get_train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
CONFUSION_DIR = FIGURES_DIR / "confusion_matrices"
MODELS_DIR = PROJECT_ROOT / "models"


def generate_eda_plots(df: pd.DataFrame) -> None:
    """Generate and save exploratory data analysis figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="label", hue="label", palette=["#d9534f", "#5cb85c"], legend=False)
    plt.xticks([0, 1], ["Fake", "Real"])
    plt.title("Target Variable Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "target_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=df,
        x="text_length",
        hue="label",
        bins=40,
        palette=["#d9534f", "#5cb85c"],
        element="step",
    )
    plt.title("Article Text Length Distribution")
    plt.xlabel("Character Count")
    plt.ylabel("Frequency")
    plt.legend(title="Label", labels=["Fake", "Real"])
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "text_length_histogram.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x="label", y="word_count", hue="label", palette=["#d9534f", "#5cb85c"], legend=False)
    plt.xticks([0, 1], ["Fake", "Real"])
    plt.title("Word Count by Label")
    plt.xlabel("Label")
    plt.ylabel("Word Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "word_count_boxplot.png", dpi=150)
    plt.close()

    numeric_df = df[["text_length", "title_length", "word_count", "label"]]
    plt.figure(figsize=(7, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        data=df.sample(min(3000, len(df)), random_state=42),
        x="title_length",
        y="text_length",
        hue="label",
        alpha=0.5,
        palette=["#d9534f", "#5cb85c"],
    )
    plt.title("Title Length vs Text Length")
    plt.xlabel("Title Length")
    plt.ylabel("Text Length")
    plt.legend(title="Label", labels=["Fake", "Real"])
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "title_vs_text_scatter.png", dpi=150)
    plt.close()


def get_positive_proba(model, x) -> list[float]:
    """Return positive-class probabilities when supported by the estimator."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1].tolist()
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x)
        return scores.tolist()
    return []


def train_and_compare(x_train, x_test, y_train, y_test) -> tuple[dict, str]:
    """Train all baseline models and return comparison metrics plus best key."""
    comparison: dict[str, dict] = {}
    builders = get_model_builders()

    for model_key, builder in builders.items():
        display_name = MODEL_DISPLAY_NAMES[model_key]
        print(f"\nTraining {display_name}...")
        model = builder()
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        y_proba = get_positive_proba(model, x_test)
        metrics = compute_metrics(y_test, y_pred, y_proba if y_proba else None)
        cv_scores = cross_validate_model(model, x_train, y_train)

        comparison[display_name] = {
            "model_key": model_key,
            **{k: v for k, v in metrics.items() if k != "classification_report"},
            **cv_scores,
        }

        plot_confusion_matrix(
            y_test,
            y_pred,
            title=f"{display_name} Confusion Matrix",
            output_path=CONFUSION_DIR / f"{model_key}.png",
        )

        if y_proba:
            plot_roc_curve(
                y_test,
                y_proba,
                title=f"{display_name} ROC Curve",
                output_path=FIGURES_DIR / f"roc_{model_key}.png",
            )

    plot_model_comparison(
        {
            name: {
                "accuracy": values["accuracy"],
                "f1": values["f1"],
                "roc_auc": values.get("roc_auc", 0.0),
            }
            for name, values in comparison.items()
        },
        output_path=FIGURES_DIR / "model_comparison.png",
    )

    best_name = max(
        comparison,
        key=lambda name: (
            comparison[name].get("f1", 0.0),
            comparison[name].get("roc_auc", 0.0),
        ),
    )
    best_key = comparison[best_name]["model_key"]
    return comparison, best_key


def main() -> None:
    """Run the full training workflow."""
    preprocessor = TextPreprocessor()
    df = preprocessor.load_dataset()
    dataset_info = preprocessor.dataset_summary(df)
    save_metrics(dataset_info, REPORTS_DIR / "dataset_info.json")

    print("Generating EDA plots...")
    generate_eda_plots(df)

    x_train, x_test, y_train, y_test = get_train_test_split(df)
    print(f"Train size: {len(x_train)} | Test size: {len(x_test)}")

    comparison, best_key = train_and_compare(x_train, x_test, y_train, y_test)
    save_metrics(comparison, REPORTS_DIR / "model_comparison.json")

    best_name = MODEL_DISPLAY_NAMES[best_key]
    print(f"\nBest baseline model: {best_name} ({best_key})")
    print("Running GridSearchCV...")
    search = optimize_model(best_key, x_train, y_train)
    best_model = search.best_estimator_

    y_pred = best_model.predict(x_test)
    y_proba = get_positive_proba(best_model, x_test)
    optimized_metrics = compute_metrics(y_test, y_pred, y_proba if y_proba else None)
    optimized_metrics["best_params"] = search.best_params_
    optimized_metrics["best_cv_score"] = float(search.best_score_)
    optimized_metrics["model_name"] = best_name
    optimized_metrics["model_key"] = best_key
    save_metrics(optimized_metrics, REPORTS_DIR / "optimized_metrics.json")

    plot_confusion_matrix(
        y_test,
        y_pred,
        title=f"Optimized {best_name} Confusion Matrix",
        output_path=FIGURES_DIR / "optimized_confusion_matrix.png",
    )
    if y_proba:
        plot_roc_curve(
            y_test,
            y_proba,
            title=f"Optimized {best_name} ROC Curve",
            output_path=FIGURES_DIR / "optimized_roc_curve.png",
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODELS_DIR / "best_model.joblib")

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_name": best_name,
        "model_key": best_key,
        "best_params": search.best_params_,
        "test_metrics": {
            key: optimized_metrics[key]
            for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]
            if key in optimized_metrics
        },
        "dataset_info": dataset_info,
    }
    with (MODELS_DIR / "training_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\nTraining complete.")
    print(f"Saved model to {MODELS_DIR / 'best_model.joblib'}")
    print(f"Optimized F1: {optimized_metrics.get('f1', 0):.4f}")


if __name__ == "__main__":
    main()
