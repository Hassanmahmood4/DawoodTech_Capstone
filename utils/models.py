"""Model definitions, training helpers, and hyperparameter tuning."""

from __future__ import annotations

from typing import Any

from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC

from utils.preprocessing import build_pipeline

MODEL_BUILDERS: dict[str, Any] = {
    "logistic_regression": lambda: build_pipeline(
        LogisticRegression(max_iter=1000, random_state=42)
    ),
    "decision_tree": lambda: build_pipeline(
        DecisionTreeClassifier(random_state=42)
    ),
    "random_forest": lambda: build_pipeline(
        RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    ),
    "svm": lambda: build_pipeline(
        CalibratedClassifierCV(LinearSVC(max_iter=3000, random_state=42), cv=3)
    ),
}

MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "svm": "Support Vector Machine",
}

PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    "logistic_regression": {
        "clf__C": [0.1, 1.0, 10.0],
        "clf__solver": ["lbfgs", "saga"],
        "clf__class_weight": [None, "balanced"],
    },
    "decision_tree": {
        "clf__max_depth": [None, 20, 40],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4],
    },
    "random_forest": {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [None, 30],
        "clf__min_samples_split": [2, 5],
        "clf__class_weight": [None, "balanced"],
    },
    "svm": {
        "clf__estimator__C": [0.1, 1.0, 10.0],
        "clf__estimator__class_weight": [None, "balanced"],
    },
}


def get_model_builders() -> dict[str, Any]:
    """Return model factory functions keyed by model name."""
    return MODEL_BUILDERS.copy()


def optimize_model(
    model_key: str,
    x_train,
    y_train,
    cv: int = 5,
    scoring: str = "f1",
    n_jobs: int = -1,
) -> GridSearchCV:
    """Run GridSearchCV for the selected model."""
    if model_key not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model key: {model_key}")

    pipeline = MODEL_BUILDERS[model_key]()
    param_grid = PARAM_GRIDS[model_key]

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        refit=True,
        verbose=1,
    )
    search.fit(x_train, y_train)
    return search
