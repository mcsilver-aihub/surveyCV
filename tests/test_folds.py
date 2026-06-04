"""Tests for design-aware fold construction."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from surveycv import design_aware_folds


def _make_nested_design(n_strata: int, psus_per_stratum: int, rows_per_psu: int):
    """Build a nested stratum/PSU design for testing.

    Args:
        n_strata: Number of strata.
        psus_per_stratum: PSUs within each stratum (labels reused across strata).
        rows_per_psu: Observations per PSU.

    Returns:
        A tuple ``(strata, clusters)`` of aligned 1-D arrays.
    """
    strata, clusters = [], []
    for s in range(n_strata):
        for p in range(psus_per_stratum):
            strata.extend([s] * rows_per_psu)
            clusters.extend([p] * rows_per_psu)  # PSU labels intentionally reused
    return np.array(strata), np.array(clusters)


def test_clusters_are_never_split_across_folds():
    """Every observation in a PSU must land in the same fold."""
    strata, clusters = _make_nested_design(4, 6, 5)
    fold_ids = design_aware_folds(strata, clusters, n_folds=3, random_state=0)

    keys = np.array([f"{s}::{c}" for s, c in zip(strata, clusters)])
    for key in np.unique(keys):
        assert len(np.unique(fold_ids[keys == key])) == 1


def test_every_fold_is_populated_and_in_range():
    """All requested folds appear and ids stay within range."""
    strata, clusters = _make_nested_design(5, 5, 4)
    n_folds = 5
    fold_ids = design_aware_folds(strata, clusters, n_folds=n_folds, random_state=1)

    assert set(np.unique(fold_ids)).issubset(set(range(n_folds)))
    assert len(np.unique(fold_ids)) == n_folds


def test_each_stratum_spans_all_folds_when_feasible():
    """With enough PSUs per stratum, each stratum touches every fold."""
    strata, clusters = _make_nested_design(3, 6, 3)
    n_folds = 3
    fold_ids = design_aware_folds(strata, clusters, n_folds=n_folds, random_state=2)

    for s in np.unique(strata):
        assert len(np.unique(fold_ids[strata == s])) == n_folds


def test_determinism_with_seed():
    """Same seed yields identical fold ids; different seeds differ."""
    strata, clusters = _make_nested_design(4, 5, 4)
    a = design_aware_folds(strata, clusters, n_folds=4, random_state=7)
    b = design_aware_folds(strata, clusters, n_folds=4, random_state=7)
    c = design_aware_folds(strata, clusters, n_folds=4, random_state=8)

    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_reused_psu_labels_are_treated_as_independent_units():
    """A PSU label reused across strata is partitioned independently per stratum.

    Both strata have PSUs labeled {0, 1}. Within each stratum those two PSUs are
    spread across the two folds, so within a stratum the same label does not
    force a shared fold; the sampling unit is the (stratum, PSU) pair.
    """
    strata = np.array([0, 0, 1, 1])
    clusters = np.array([0, 1, 0, 1])
    fold_ids = design_aware_folds(strata, clusters, n_folds=2, random_state=0)

    for s in (0, 1):
        assert len(np.unique(fold_ids[strata == s])) == 2


def test_stratified_only_design_balances_folds():
    """A stratified design without clusters still spreads each stratum."""
    strata = np.repeat([0, 1, 2], 9)
    fold_ids = design_aware_folds(strata, clusters=None, n_folds=3, random_state=3)

    for s in np.unique(strata):
        assert len(np.unique(fold_ids[strata == s])) == 3


def test_warns_when_stratum_has_too_few_psus():
    """A stratum with fewer PSUs than folds triggers a feasibility warning."""
    strata = np.array([0, 0, 0, 1, 1, 1])
    clusters = np.array([0, 1, 2, 0, 0, 0])  # stratum 1 has a single PSU
    with pytest.warns(UserWarning, match="fewer than n_folds"):
        design_aware_folds(strata, clusters, n_folds=3, random_state=0)


def test_requires_at_least_one_design_dimension():
    """Both strata and clusters None is an error."""
    with pytest.raises(ValueError, match="At least one of strata or clusters"):
        design_aware_folds(strata=None, clusters=None, n_folds=3)


def test_rejects_invalid_fold_count():
    """n_folds below the minimum is rejected."""
    strata, clusters = _make_nested_design(3, 4, 2)
    with pytest.raises(ValueError, match="at least"):
        design_aware_folds(strata, clusters, n_folds=1)


def test_rejects_misaligned_lengths():
    """Design arrays of differing length are rejected."""
    with pytest.raises(ValueError, match="length"):
        design_aware_folds(np.array([0, 1, 2]), np.array([0, 1]), n_folds=2)


def test_no_warning_when_feasible():
    """A well-sized design produces no feasibility warning."""
    strata, clusters = _make_nested_design(3, 5, 3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        design_aware_folds(strata, clusters, n_folds=3, random_state=0)
