"""Tests for design-aware, survey-weighted cross-validation scoring."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from surveycv import SurveyFold, cross_val_score_survey


def _design(n_strata: int = 4, psus: int = 6, rows: int = 8):
    """Return aligned (strata, clusters, X, y, weights) for a nested design."""
    strata, clusters = [], []
    for s in range(n_strata):
        for p in range(psus):
            strata.extend([s] * rows)
            clusters.extend([p] * rows)
    strata = np.array(strata)
    clusters = np.array(clusters)
    n = len(strata)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 4))
    # Signal so AUC is meaningfully above chance.
    y = (X[:, 0] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    weights = rng.uniform(0.5, 3.0, size=n)
    return strata, clusters, X, y, weights


def test_returns_one_score_per_fold():
    """The driver returns exactly n_splits scores."""
    strata, clusters, X, y, weights = _design()
    cv = SurveyFold(n_splits=3, strata=strata, clusters=clusters, random_state=0)
    scores = cross_val_score_survey(
        LogisticRegression(max_iter=200), X, y, cv=cv, weights=weights, scoring="roc_auc"
    )
    assert scores.shape == (3,)


def test_unweighted_matches_manual_roc_auc():
    """With weights=None, the per-fold score equals a manual unweighted AUC."""
    strata, clusters, X, y, _ = _design()
    cv = SurveyFold(n_splits=3, strata=strata, clusters=clusters, random_state=1)

    scores = cross_val_score_survey(
        LogisticRegression(max_iter=200), X, y, cv=cv, weights=None, scoring="roc_auc"
    )

    manual = []
    for train_idx, test_idx in cv.split(X):
        model = LogisticRegression(max_iter=200).fit(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[test_idx])[:, 1]
        manual.append(roc_auc_score(y[test_idx], proba))
    assert np.allclose(scores, manual)


def test_weighted_score_uses_correct_weight_slice():
    """Weighted scoring matches a manual Horvitz-Thompson weighted AUC per fold."""
    strata, clusters, X, y, weights = _design()
    cv = SurveyFold(n_splits=3, strata=strata, clusters=clusters, random_state=2)

    scores = cross_val_score_survey(
        LogisticRegression(max_iter=200), X, y, cv=cv, weights=weights, scoring="roc_auc"
    )

    manual = []
    for train_idx, test_idx in cv.split(X):
        model = LogisticRegression(max_iter=200).fit(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[test_idx])[:, 1]
        manual.append(roc_auc_score(y[test_idx], proba, sample_weight=weights[test_idx]))
    assert np.allclose(scores, manual)


def test_neg_mse_is_negated_so_higher_is_better():
    """neg_mean_squared_error returns non-positive scores."""
    strata, clusters, X, y, weights = _design()
    cv = SurveyFold(n_splits=3, strata=strata, clusters=clusters, random_state=3)
    scores = cross_val_score_survey(
        LogisticRegression(max_iter=200),
        X,
        y,
        cv=cv,
        weights=weights,
        scoring="neg_mean_squared_error",
    )
    assert np.all(scores <= 0)


def test_unsupported_scoring_raises():
    """An unknown scoring name is rejected."""
    strata, clusters, X, y, _ = _design()
    cv = SurveyFold(n_splits=3, strata=strata, clusters=clusters, random_state=0)
    with pytest.raises(ValueError, match="Unsupported scoring"):
        cross_val_score_survey(LogisticRegression(max_iter=200), X, y, cv=cv, scoring="f1_made_up")


def test_mismatched_weight_length_raises():
    """Weights of the wrong length are rejected."""
    strata, clusters, X, y, _ = _design()
    cv = SurveyFold(n_splits=3, strata=strata, clusters=clusters, random_state=0)
    with pytest.raises(ValueError, match="length"):
        cross_val_score_survey(LogisticRegression(max_iter=200), X, y, cv=cv, weights=np.ones(3))


def test_works_with_pandas_dataframe():
    """A pandas DataFrame X is handled via positional row selection."""
    pd = pytest.importorskip("pandas")
    strata, clusters, X, y, weights = _design()
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    cv = SurveyFold(n_splits=3, strata=strata, clusters=clusters, random_state=0)
    scores = cross_val_score_survey(
        LogisticRegression(max_iter=200), X_df, y, cv=cv, weights=weights
    )
    assert scores.shape == (3,)
