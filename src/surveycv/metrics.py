"""Survey-weighted evaluation metrics.

Small helpers for the weighted statistics that come up when evaluating a model on
complex survey data: weighted prevalence, sensitivity, specificity, and AUC, plus
the general weighted mean they are built on.

Each function takes full arrays and is written to drop straight into
:func:`cluster_bootstrap_ci` as the ``statistic`` callable, for example::

    lambda idx: weighted_sensitivity(y[idx], pred[idx], w[idx])

When a metric is undefined on the given data (a resample with no positive cases,
a single outcome class, or zero total weight) it returns ``nan`` rather than
raising. That is exactly what the cluster bootstrap expects, so a few degenerate
resamples on a rare outcome do not break the confidence interval.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def _aligned_1d(*arrays_with_names: tuple) -> list[np.ndarray]:
    """Coerce inputs to aligned 1-D arrays.

    Args:
        *arrays_with_names: One or more ``(array_like, name)`` pairs, where
            ``name`` is used in error messages.

    Returns:
        A list of 1-D numpy arrays in the order given.

    Raises:
        ValueError: If any input is not one-dimensional, or the inputs differ in
            length.
    """
    arrays: list[np.ndarray] = []
    n: int | None = None
    for values, name in arrays_with_names:
        array = np.asarray(values)
        if array.ndim != 1:
            raise ValueError(f"{name} must be 1-dimensional, got shape {array.shape}.")
        if n is None:
            n = len(array)
        elif len(array) != n:
            raise ValueError(f"{name} has length {len(array)} but {n} was expected.")
        arrays.append(array)
    return arrays


def weighted_mean(values: object, weights: object) -> float:
    """Compute the survey-weighted mean of ``values``.

    Args:
        values: 1-D array of numeric values.
        weights: 1-D array of survey weights, aligned with ``values``.

    Returns:
        The weighted mean ``sum(values * weights) / sum(weights)``, or ``nan`` if
        the total weight is not positive.

    Raises:
        ValueError: If the arrays are not 1-D or differ in length.

    Edge cases:
        An empty input or all-zero weights yields ``nan`` instead of raising.
    """
    v, w = _aligned_1d((values, "values"), (weights, "weights"))
    w = w.astype(float)
    total = w.sum()
    if total <= 0:
        return float("nan")
    return float(np.dot(v.astype(float), w) / total)


def weighted_prevalence(y_true: object, weights: object) -> float:
    """Compute the survey-weighted prevalence of a binary outcome.

    Args:
        y_true: 1-D array of 0/1 outcomes.
        weights: 1-D array of survey weights, aligned with ``y_true``.

    Returns:
        The weighted share of positives, or ``nan`` if the total weight is not
        positive.

    Raises:
        ValueError: If the arrays are not 1-D or differ in length.
    """
    return weighted_mean(y_true, weights)


def weighted_sensitivity(y_true: object, y_pred: object, weights: object) -> float:
    """Compute survey-weighted sensitivity (true positive rate).

    Among the true positives, this is the weighted fraction the model flagged.

    Args:
        y_true: 1-D array of 0/1 true outcomes.
        y_pred: 1-D array of 0/1 predicted labels (already thresholded), aligned
            with ``y_true``.
        weights: 1-D array of survey weights, aligned with ``y_true``.

    Returns:
        The weighted true positive rate, or ``nan`` if there are no positive
        cases (or their total weight is zero).

    Raises:
        ValueError: If the arrays are not 1-D or differ in length.

    Edge cases:
        Negatives are ignored. A resample with no positives yields ``nan``.
    """
    y, pred, w = _aligned_1d((y_true, "y_true"), (y_pred, "y_pred"), (weights, "weights"))
    positives = y == 1
    if not positives.any():
        return float("nan")
    return weighted_mean(pred[positives], w[positives])


def weighted_specificity(y_true: object, y_pred: object, weights: object) -> float:
    """Compute survey-weighted specificity (true negative rate).

    Among the true negatives, this is the weighted fraction the model did not
    flag.

    Args:
        y_true: 1-D array of 0/1 true outcomes.
        y_pred: 1-D array of 0/1 predicted labels (already thresholded), aligned
            with ``y_true``.
        weights: 1-D array of survey weights, aligned with ``y_true``.

    Returns:
        The weighted true negative rate, or ``nan`` if there are no negative
        cases (or their total weight is zero).

    Raises:
        ValueError: If the arrays are not 1-D or differ in length.

    Edge cases:
        Positives are ignored. A resample with no negatives yields ``nan``.
    """
    y, pred, w = _aligned_1d((y_true, "y_true"), (y_pred, "y_pred"), (weights, "weights"))
    negatives = y == 0
    if not negatives.any():
        return float("nan")
    predicted_negative = (pred[negatives] == 0).astype(float)
    return weighted_mean(predicted_negative, w[negatives])


def weighted_auc(y_true: object, y_score: object, weights: object) -> float:
    """Compute the survey-weighted ROC AUC.

    Wraps :func:`sklearn.metrics.roc_auc_score` with ``sample_weight``.

    Args:
        y_true: 1-D array of 0/1 true outcomes.
        y_score: 1-D array of predicted scores or probabilities, aligned with
            ``y_true``.
        weights: 1-D array of survey weights, aligned with ``y_true``.

    Returns:
        The weighted AUC, or ``nan`` if only one outcome class is present (where
        AUC is undefined).

    Raises:
        ValueError: If the arrays are not 1-D or differ in length.

    Edge cases:
        Returning ``nan`` for a single-class input lets the function be used
        directly inside a cluster bootstrap on a rare outcome.
    """
    y, score, w = _aligned_1d((y_true, "y_true"), (y_score, "y_score"), (weights, "weights"))
    if y.size == 0 or y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, score, sample_weight=w.astype(float)))
