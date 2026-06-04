"""Tests for the design-aware train/test split."""

from __future__ import annotations

import numpy as np
import pytest

from surveycv import survey_train_test_split


def _nested_design(n_strata: int = 5, psus: int = 10, rows: int = 4):
    """Return aligned (strata, clusters) for a nested design."""
    strata, clusters = [], []
    for s in range(n_strata):
        for p in range(psus):
            strata.extend([s] * rows)
            clusters.extend([p] * rows)
    return np.array(strata), np.array(clusters)


def test_split_partitions_all_rows():
    """Train and test indices are disjoint and cover every row."""
    strata, clusters = _nested_design()
    train_idx, test_idx = survey_train_test_split(strata, clusters, test_size=0.2, random_state=0)
    n = len(strata)
    assert set(train_idx).isdisjoint(set(test_idx))
    assert set(train_idx) | set(test_idx) == set(range(n))


def test_no_psu_leaks_across_split():
    """A PSU lands wholly in train or wholly in test."""
    strata, clusters = _nested_design()
    train_idx, test_idx = survey_train_test_split(strata, clusters, test_size=0.3, random_state=2)
    keys = np.array([f"{s}::{c}" for s, c in zip(strata, clusters)])
    assert set(keys[train_idx]).isdisjoint(set(keys[test_idx]))


def test_every_stratum_represented_in_test_when_feasible():
    """With multiple PSUs per stratum, each stratum appears in the test set."""
    strata, clusters = _nested_design(n_strata=5, psus=10, rows=3)
    _, test_idx = survey_train_test_split(strata, clusters, test_size=0.2, random_state=1)
    assert set(np.unique(strata[test_idx])) == set(np.unique(strata))


def test_holdout_fraction_is_approximately_respected():
    """Held-out cluster share is close to the requested test_size."""
    strata, clusters = _nested_design(n_strata=4, psus=10, rows=5)
    keys = np.array([f"{s}::{c}" for s, c in zip(strata, clusters)])
    _, test_idx = survey_train_test_split(strata, clusters, test_size=0.2, random_state=3)
    held_out = len(np.unique(keys[test_idx]))
    total = len(np.unique(keys))
    assert abs(held_out / total - 0.2) < 0.1


def test_determinism_with_seed():
    """Same seed yields the same split."""
    strata, clusters = _nested_design()
    a = survey_train_test_split(strata, clusters, test_size=0.25, random_state=5)
    b = survey_train_test_split(strata, clusters, test_size=0.25, random_state=5)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])


def test_singleton_stratum_warns_and_stays_in_train():
    """A single-cluster stratum cannot enter test and triggers a warning."""
    strata = np.array([0, 0, 0, 0, 1, 1])
    clusters = np.array([0, 1, 2, 3, 0, 0])  # stratum 1 has one PSU
    with pytest.warns(UserWarning, match="only one cluster"):
        _, test_idx = survey_train_test_split(strata, clusters, test_size=0.5, random_state=0)
    assert 1 not in set(strata[test_idx])


def test_requires_clusters():
    """Omitting clusters is an error for a design-aware split."""
    with pytest.raises(ValueError, match="clusters must be provided"):
        survey_train_test_split(strata=np.array([0, 1]), clusters=None)


def test_rejects_out_of_range_test_size():
    """test_size outside (0, 1) is rejected."""
    strata, clusters = _nested_design()
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        survey_train_test_split(strata, clusters, test_size=1.0)
