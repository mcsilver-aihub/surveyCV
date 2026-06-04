"""Survey-weighted cross-validation scoring.

Wieczorek et al. (2022) make two recommendations: build folds that respect the
design, and evaluate the test-fold loss with survey weights so the chosen model
is optimized against a population-representative objective rather than the
unweighted sample mean. This module supplies the weighted scoring half.

The :func:`cross_val_score_survey` driver loops folds explicitly and applies the
correct weight slice to each test fold. This sidesteps scikit-learn's
``sample_weight`` metadata routing, which does not reliably thread per-fold test
weights through ``cross_val_score`` across versions.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    roc_auc_score,
)

from .splitter import SurveyFold

Metric = Callable[..., float]

_WEIGHTED_METRICS: dict = {
    "roc_auc": roc_auc_score,
    "accuracy": accuracy_score,
    "neg_mean_squared_error": mean_squared_error,
}

_PROBABILITY_METRICS = frozenset({"roc_auc"})
_NEGATED_METRICS = frozenset({"neg_mean_squared_error"})


def _resolve_metric(scoring: str) -> Metric:
    """Look up a supported weighted metric by name.

    Args:
        scoring: Name of the metric. One of the keys of ``_WEIGHTED_METRICS``.

    Returns:
        The scikit-learn metric callable for ``scoring``.

    Raises:
        ValueError: If ``scoring`` is not a supported metric name.
    """
    if scoring not in _WEIGHTED_METRICS:
        supported = ", ".join(sorted(_WEIGHTED_METRICS))
        raise ValueError(f"Unsupported scoring '{scoring}'. Supported: {supported}.")
    return _WEIGHTED_METRICS[scoring]


def _predict_for_metric(estimator: object, X_test: object, scoring: str) -> np.ndarray:
    """Produce the predictions a given metric needs.

    Args:
        estimator: A fitted estimator.
        X_test: Test-fold features.
        scoring: Metric name, used to decide between probabilities and labels.

    Returns:
        Predicted positive-class probabilities for probability metrics, else
        predicted labels.

    Raises:
        AttributeError: If a probability metric is requested but the estimator
            lacks ``predict_proba``.
    """
    if scoring in _PROBABILITY_METRICS:
        proba = estimator.predict_proba(X_test)  # type: ignore[attr-defined]
        return np.asarray(proba)[:, 1]
    return estimator.predict(X_test)  # type: ignore[attr-defined]


def _score_one_fold(
    estimator: object,
    X: object,
    y: np.ndarray,
    weights: np.ndarray | None,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    scoring: str,
) -> float:
    """Fit on the train fold and return the weighted score on the test fold.

    Args:
        estimator: Unfitted estimator; cloned so the caller's object is untouched.
        X: Full feature matrix (numpy array or pandas DataFrame).
        y: Full target array.
        weights: Full survey-weight array, or ``None`` for unweighted scoring.
        train_idx: Row indices for the training fold.
        test_idx: Row indices for the test fold.
        scoring: Metric name.

    Returns:
        The (possibly negated) weighted metric value for this fold.

    Edge cases:
        Negated metrics (e.g. ``neg_mean_squared_error``) are returned negated so
        that higher is always better, matching scikit-learn's scorer convention.
    """
    model = clone(estimator)
    X_train, X_test = _take_rows(X, train_idx), _take_rows(X, test_idx)
    model.fit(X_train, y[train_idx])  # type: ignore[attr-defined]

    y_pred = _predict_for_metric(model, X_test, scoring)
    sample_weight = None if weights is None else weights[test_idx]
    metric = _resolve_metric(scoring)
    value = float(metric(y[test_idx], y_pred, sample_weight=sample_weight))
    return -value if scoring in _NEGATED_METRICS else value


def cross_val_score_survey(
    estimator: object,
    X: object,
    y: object,
    cv: SurveyFold,
    weights: object = None,
    scoring: str = "roc_auc",
) -> np.ndarray:
    """Cross-validate an estimator with design-aware folds and weighted scoring.

    Each fold is held out in turn (per the design-aware ``cv``), the estimator is
    fit on the remaining folds, and the test-fold metric is computed using the
    survey weights of the held-out rows. Use this as an Optuna objective for a
    survey-correct hyperparameter search.

    Args:
        estimator: Any scikit-learn-compatible estimator (e.g. an XGBoost
            classifier using the sklearn API). It is cloned per fold.
        X: Feature matrix aligned with the design labels in ``cv``.
        y: Target array of length ``n_rows``.
        cv: A :class:`SurveyFold` carrying the stratum/cluster design.
        weights: Survey weights of length ``n_rows``, or ``None`` for unweighted
            scoring (folds remain design-aware regardless).
        scoring: Metric name; one of ``roc_auc``, ``accuracy``,
            ``neg_mean_squared_error``.

    Returns:
        A 1-D array of per-fold scores (higher is better) of length
        ``cv.get_n_splits()``.

    Raises:
        ValueError: If ``scoring`` is unsupported or array lengths disagree.

    Edge cases:
        ``weights`` of ``None`` yields ordinary (unweighted) metrics while still
        respecting the survey design in fold construction.
    """
    y_arr = np.asarray(y)
    weights_arr = None if weights is None else np.asarray(weights, dtype=float)
    if weights_arr is not None and len(weights_arr) != len(y_arr):
        raise ValueError(f"weights has length {len(weights_arr)} but y has length {len(y_arr)}.")

    scores = [
        _score_one_fold(estimator, X, y_arr, weights_arr, train_idx, test_idx, scoring)
        for train_idx, test_idx in cv.split(X)
    ]
    return np.array(scores)


def _take_rows(X: object, idx: np.ndarray) -> object:
    """Select rows from a feature container by integer position.

    Args:
        X: A numpy array or pandas DataFrame.
        idx: Integer row indices to select.

    Returns:
        The selected rows, preserving the container type (DataFrame stays a
        DataFrame via ``iloc``; ndarray stays an ndarray).
    """
    if hasattr(X, "iloc"):
        return X.iloc[idx]  # type: ignore[attr-defined]
    return np.asarray(X)[idx]
