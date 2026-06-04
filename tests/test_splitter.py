"""Tests for the scikit-learn-compatible SurveyFold cross-validator."""

from __future__ import annotations

import numpy as np
import pytest

from surveycv import SurveyFold


def _design(n_strata: int = 4, psus: int = 6, rows: int = 5):
    """Return aligned (strata, clusters, X, y) for a nested design."""
    strata, clusters = [], []
    for s in range(n_strata):
        for p in range(psus):
            strata.extend([s] * rows)
            clusters.extend([p] * rows)
    strata = np.array(strata)
    clusters = np.array(clusters)
    n = len(strata)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 3))
    y = rng.integers(0, 2, size=n)
    return strata, clusters, X, y


def test_get_n_splits_matches_config():
    """get_n_splits returns the configured number of folds."""
    cv = SurveyFold(n_splits=4, strata=np.array([0, 1]), clusters=np.array([0, 1]))
    assert cv.get_n_splits() == 4


def test_split_yields_expected_number_of_folds():
    """split yields exactly n_splits train/test pairs."""
    strata, clusters, X, _ = _design()
    cv = SurveyFold(n_splits=3, strata=strata, clusters=clusters, random_state=0)
    assert len(list(cv.split(X))) == 3


def test_train_test_are_disjoint_and_cover_all_rows():
    """Each fold's train and test indices partition the row set."""
    strata, clusters, X, _ = _design()
    cv = SurveyFold(n_splits=3, strata=strata, clusters=clusters, random_state=0)
    n = len(X)
    for train_idx, test_idx in cv.split(X):
        assert set(train_idx).isdisjoint(set(test_idx))
        assert set(train_idx) | set(test_idx) == set(range(n))


def test_no_psu_appears_in_both_train_and_test():
    """A PSU is wholly in train or wholly in test, never both."""
    strata, clusters, X, _ = _design()
    cv = SurveyFold(n_splits=3, strata=strata, clusters=clusters, random_state=1)
    keys = np.array([f"{s}::{c}" for s, c in zip(strata, clusters)])
    for train_idx, test_idx in cv.split(X):
        assert set(keys[train_idx]).isdisjoint(set(keys[test_idx]))


def test_misaligned_X_raises():
    """Passing X whose length disagrees with the design labels raises."""
    strata, clusters, _, _ = _design()
    cv = SurveyFold(n_splits=3, strata=strata, clusters=clusters)
    with pytest.raises(ValueError, match="rows"):
        list(cv.split(np.zeros((3, 2))))


def test_works_with_sklearn_cross_val_score():
    """SurveyFold drops into sklearn's cross_val_score as a cv object."""
    sklearn = pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    strata, clusters, X, y = _design()
    cv = SurveyFold(n_splits=3, strata=strata, clusters=clusters, random_state=0)
    scores = cross_val_score(LogisticRegression(max_iter=200), X, y, cv=cv)
    assert len(scores) == 3
    assert sklearn is not None
