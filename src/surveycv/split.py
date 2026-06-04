"""Design-aware train/test splitting for complex sample surveys.

Provides a single held-out split (rather than K folds) in which whole clusters
are assigned to either train or test. This is the split a survey-based
prediction model should use for its primary evaluation: students from the same
school (PSU) never appear in both train and test, which prevents the model from
memorizing school-specific patterns and inflating performance.
"""

from __future__ import annotations

import warnings

import numpy as np

from ._validation import (
    align_lengths,
    cluster_keys_within_strata,
    require_design,
    to_1d_array,
)

_MIN_TEST_SIZE = 0.0
_MAX_TEST_SIZE = 1.0


def _validate_test_size(test_size: float) -> None:
    """Validate that the test fraction lies strictly inside the unit interval.

    Args:
        test_size: Requested fraction of clusters to hold out for testing.

    Raises:
        ValueError: If ``test_size`` is not strictly between 0 and 1.
    """
    if not _MIN_TEST_SIZE < test_size < _MAX_TEST_SIZE:
        raise ValueError(f"test_size must be strictly between 0 and 1, got {test_size}.")


def _holdout_count(n_clusters: int, test_size: float) -> int:
    """Choose how many clusters to hold out from a stratum.

    Guarantees at least one cluster in both train and test whenever the stratum
    has two or more clusters, so no stratum is lost from either split.

    Args:
        n_clusters: Number of clusters available in the stratum.
        test_size: Requested test fraction.

    Returns:
        The number of clusters to place in the test set, in ``[0, n_clusters]``.

    Edge cases:
        With a single-cluster stratum, returns 0 (the lone cluster stays in
        train) and the caller is expected to warn, because that stratum cannot
        be represented in the test set.
    """
    if n_clusters < 2:
        return 0
    n_test = max(1, int(round(test_size * n_clusters)))
    return min(n_clusters - 1, n_test)


def survey_train_test_split(
    strata: object = None,
    clusters: object = None,
    test_size: float = 0.2,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Split rows into train/test by holding out whole clusters within strata.

    Within each stratum a fraction of the clusters is assigned to the test set
    and the rest to train, so the test set stays representative of every stratum
    while keeping each cluster wholly on one side of the split.

    Args:
        strata: Array-like of stratum labels, or ``None`` for an unstratified
            design (clusters are then split across the whole sample).
        clusters: Array-like of cluster (PSU) labels. Required; without clusters
            there is nothing to hold out as a group, so use scikit-learn's
            ``train_test_split`` instead.
        test_size: Fraction of clusters (within each stratum) to hold out for
            testing. Defaults to 0.2.
        random_state: Seed for a reproducible split, or ``None``.

    Returns:
        A tuple ``(train_index, test_index)`` of integer numpy arrays giving the
        row positions for each split.

    Raises:
        ValueError: If ``clusters`` is ``None``, if ``test_size`` is out of
            range, or if the design arrays differ in length.

    Edge cases:
        Strata with a single cluster cannot appear in the test set; a warning is
        emitted listing how many strata were affected, and their rows remain in
        the training set.
    """
    _validate_test_size(test_size)

    if clusters is None:
        raise ValueError(
            "clusters must be provided for a design-aware split; for simple "
            "random sampling use sklearn.model_selection.train_test_split."
        )

    strata_arr = to_1d_array(strata, "strata") if strata is not None else None
    clusters_arr = to_1d_array(clusters, "clusters")
    require_design(strata_arr, clusters_arr)

    n_rows = len(clusters_arr)
    align_lengths(n_rows, strata_arr, clusters_arr)

    cluster_keys = cluster_keys_within_strata(strata_arr, clusters_arr)
    rng = np.random.default_rng(random_state)

    test_clusters: set = set()
    n_singleton_strata = 0
    group_labels = np.unique(strata_arr) if strata_arr is not None else np.array([None])

    for group in group_labels:
        group_mask = np.ones(n_rows, dtype=bool) if strata_arr is None else strata_arr == group
        group_clusters = np.unique(cluster_keys[group_mask])
        n_test = _holdout_count(len(group_clusters), test_size)
        if len(group_clusters) < 2:
            n_singleton_strata += 1
        if n_test == 0:
            continue
        shuffled = group_clusters.copy()
        rng.shuffle(shuffled)
        test_clusters.update(shuffled[:n_test].tolist())

    if n_singleton_strata > 0:
        warnings.warn(
            f"{n_singleton_strata} stratum/strata have only one cluster and "
            "cannot be represented in the test set; their rows stay in train.",
            stacklevel=2,
        )

    test_mask = np.array([key in test_clusters for key in cluster_keys])
    all_index = np.arange(n_rows)
    return all_index[~test_mask], all_index[test_mask]
