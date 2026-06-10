"""Tests for the survey-weighted metrics."""

from __future__ import annotations

import numpy as np
import pytest

from surveycv import (
    weighted_auc,
    weighted_mean,
    weighted_prevalence,
    weighted_sensitivity,
    weighted_specificity,
)


def test_weighted_mean_matches_numpy_average():
    """weighted_mean equals numpy's weighted average."""
    v = [1.0, 2.0, 3.0, 4.0]
    w = [4.0, 1.0, 1.0, 2.0]
    assert weighted_mean(v, w) == pytest.approx(np.average(v, weights=w))


def test_weighted_prevalence_worked_example():
    """The documented prevalence example: 6/10 = 0.60 vs unweighted 0.40."""
    y = [1, 0, 0, 1, 0]
    w = [4, 1, 1, 2, 2]
    assert weighted_prevalence(y, w) == pytest.approx(0.60)
    assert np.mean(y) == pytest.approx(0.40)


def test_weighted_sensitivity_worked_example():
    """The documented sensitivity example: 7/10 = 0.70 vs unweighted 0.50."""
    # Four true attempters; the two flagged ones carry more weight.
    y_true = [1, 1, 1, 1]
    y_pred = [1, 1, 0, 0]
    w = [4, 3, 2, 1]
    assert weighted_sensitivity(y_true, y_pred, w) == pytest.approx(0.70)
    assert np.mean(y_pred) == pytest.approx(0.50)


def test_weighted_sensitivity_ignores_negatives():
    """Negatives do not affect sensitivity."""
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 1, 1]  # the negatives are predicted positive but must not count
    w = [2, 2, 5, 5]
    # Among positives: weights 2,2; flags 1,0 -> 2/4 = 0.5
    assert weighted_sensitivity(y_true, y_pred, w) == pytest.approx(0.5)


def test_weighted_specificity_basic():
    """Specificity is the weighted fraction of negatives predicted negative."""
    y_true = [0, 0, 0, 1]
    y_pred = [0, 0, 1, 1]  # among negatives: predicted-negative flags 1,1,0
    w = [3, 1, 2, 9]
    # negatives weights 3,1,2; predicted-negative 1,1,0 -> (3+1)/(3+1+2) = 4/6
    assert weighted_specificity(y_true, y_pred, w) == pytest.approx(4 / 6)


def test_weighted_auc_worked_example():
    """The documented AUC example: weighted 0.833 vs unweighted 0.75."""
    y = [1, 1, 0, 0]
    score = [0.80, 0.40, 0.60, 0.20]
    w = [3, 1, 2, 1]
    assert weighted_auc(y, score, w) == pytest.approx(0.8333, abs=1e-4)


def test_undefined_metrics_return_nan():
    """Degenerate inputs return nan rather than raising (bootstrap-friendly)."""
    # No positives -> sensitivity undefined.
    assert np.isnan(weighted_sensitivity([0, 0], [0, 0], [1, 1]))
    # No negatives -> specificity undefined.
    assert np.isnan(weighted_specificity([1, 1], [1, 1], [1, 1]))
    # One class only -> AUC undefined.
    assert np.isnan(weighted_auc([1, 1, 1], [0.2, 0.5, 0.9], [1, 1, 1]))
    # Zero total weight -> mean undefined.
    assert np.isnan(weighted_mean([1, 2, 3], [0, 0, 0]))


def test_mismatched_lengths_raise():
    """Arrays of differing length raise a clear error."""
    with pytest.raises(ValueError, match="length"):
        weighted_prevalence([1, 0, 1], [1, 1])


def test_usable_inside_cluster_bootstrap():
    """The metrics compose with cluster_bootstrap_ci as statistic callables."""
    from surveycv import cluster_bootstrap_ci

    rng = np.random.default_rng(0)
    strata, clusters, y, w = [], [], [], []
    for s in range(6):
        for p in range(6):
            for _ in range(20):
                strata.append(s)
                clusters.append(p)
                y.append(int(rng.random() < 0.2))
                w.append(rng.uniform(0.5, 3.0))
    y = np.array(y)
    w = np.array(w, dtype=float)

    res = cluster_bootstrap_ci(
        lambda idx: weighted_prevalence(y[idx], w[idx]),
        clusters=np.array(clusters),
        strata=np.array(strata),
        n_boot=200,
        random_state=0,
    )
    assert res.ci_low <= res.estimate <= res.ci_high
