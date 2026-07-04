"""Shared utilities for the Fake News Prediction project."""

from utils.preprocessing import (
    TextPreprocessor,
    build_pipeline,
    clean_text,
    get_train_test_split,
    load_dataset,
)
from utils.models import MODEL_BUILDERS, get_model_builders, optimize_model
from utils.evaluation import (
    compute_metrics,
    cross_validate_model,
    plot_confusion_matrix,
    plot_model_comparison,
    save_metrics,
)

__all__ = [
    "TextPreprocessor",
    "build_pipeline",
    "clean_text",
    "get_train_test_split",
    "load_dataset",
    "MODEL_BUILDERS",
    "get_model_builders",
    "optimize_model",
    "compute_metrics",
    "cross_validate_model",
    "plot_confusion_matrix",
    "plot_model_comparison",
    "save_metrics",
]
