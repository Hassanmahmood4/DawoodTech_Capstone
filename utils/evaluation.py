"""Evaluation metrics, cross-validation, and visualization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_validate

LABEL_NAMES = {0: "Fake", 1: "Real"}


def compute_metrics(y_true, y_pred, y_proba=None) -> dict[str, Any]:
    """Compute classification metrics for binary fake/real labels."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=["Fake", "Real"],
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    if y_proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    return metrics


def cross_validate_model(model, x_train, y_train, cv: int = 5) -> dict[str, float]:
    """Run cross-validation and return mean/std for key metrics."""
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    scores = cross_validate(
        model,
        x_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )
    return {
        f"cv_{metric}_mean": float(scores[f"test_{metric}"].mean())
        for metric in scoring
    } | {
        f"cv_{metric}_std": float(scores[f"test_{metric}"].std())
        for metric in scoring
    }


def plot_confusion_matrix(
    y_true,
    y_pred,
    title: str,
    output_path: Path,
) -> None:
    """Save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Fake", "Real"],
        yticklabels=["Fake", "Real"],
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_roc_curve(
    y_true,
    y_proba,
    title: str,
    output_path: Path,
) -> None:
    """Save an ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_model_comparison(
    comparison: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    """Create a grouped bar chart comparing model metrics."""
    metrics = ["accuracy", "f1", "roc_auc"]
    model_names = list(comparison.keys())
    x = np.arange(len(model_names))
    width = 0.22

    plt.figure(figsize=(10, 6))
    for index, metric in enumerate(metrics):
        values = [comparison[name].get(metric, 0.0) for name in model_names]
        plt.bar(x + index * width, values, width=width, label=metric.upper())

    plt.xticks(x + width, model_names, rotation=15, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Model Performance Comparison")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_metrics(payload: dict[str, Any], output_path: Path) -> None:
    """Persist metrics JSON for the Streamlit app and documentation."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def get_prediction_details(model, text: str) -> dict[str, Any]:
    """Return label, confidence, and class probabilities for one article."""
    prediction = int(model.predict([text])[0])
    label = LABEL_NAMES[prediction]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([text])[0]
        confidence = float(probabilities[prediction])
        prob_fake = float(probabilities[0])
        prob_real = float(probabilities[1])
    else:
        confidence = None
        prob_fake = None
        prob_real = None

    return {
        "prediction": prediction,
        "label": label,
        "confidence": confidence,
        "prob_fake": prob_fake,
        "prob_real": prob_real,
    }
