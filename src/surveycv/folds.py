"""Design-aware fold construction for complex sample surveys.

Clean-room implementation of the fold-assignment rules from Wieczorek, Guerin
and McMahon (2022), "K-fold cross-validation for complex sample surveys",
Stat 11(1):e454. The governing principle: cross-validation folds must mirror
the sampling design, otherwise prediction-error estimates and model selection
are biased.

Rules:
    * Cluster (PSU) sampling: every observation in a cluster goes to one fold.
    * Stratified sampling: folds are balanced across strata so each fold draws
      from every stratum.
    * Nested clustered-and-stratified (YRBS, NSFG): each stratum's clusters are
      partitioned across folds independently.
"""

from __future__ import annotations

import numpy as np

from ._validation import (
    align_lengths,
    check_n_folds,
    cluster_keys_within_strata,
    require_design,
    to_1d_array,
    warn_if_infeasible,
)


def _assign_units_to_folds(units: np.ndarray, n_folds: int, rng: np.random.Generator) -> dict:
    """Partition sampling units as evenly as possible across folds.

    Args:
        units: 1-D array of unique sampling-unit labels (cluster keys, or row
            indices when the design has no clusters).
        n_folds: Number of folds to spread the units across.
        rng: Seeded random generator controlling the shuffle.

    Returns:
        A dict mapping each unit label to its integer fold index in
        ``range(n_folds)``.

    Edge cases:
        When there are fewer units than folds, the highest fold indices receive
        no unit from this call; balance across the whole sample is still kept
        because each stratum is assigned independently.
    """
    shuffled = units.copy()
    rng.shuffle(shuffled)
    fold_of_unit = np.arange(len(shuffled)) % n_folds
    return dict(zip(shuffled.tolist(), fold_of_unit.tolist()))


def _folds_clustered(
    strata: np.ndarray | None,
    cluster_keys: np.ndarray,
    n_folds: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Assign folds when clusters are present, keeping each cluster intact.

    Args:
        strata: Stratum labels, or ``None`` if unstratified. When present,
            clusters are partitioned within each stratum independently.
        cluster_keys: Per-row cluster keys (nested within strata).
        n_folds: Number of folds.
        rng: Seeded random generator.

    Returns:
        A 1-D integer array of fold indices, one per row.
    """
    fold_ids = np.full(len(cluster_keys), -1, dtype=int)

    if strata is None:
        mapping = _assign_units_to_folds(np.unique(cluster_keys), n_folds, rng)
        for i, key in enumerate(cluster_keys):
            fold_ids[i] = mapping[key]
        return fold_ids

    for stratum in np.unique(strata):
        stratum_mask = strata == stratum
        stratum_clusters = np.unique(cluster_keys[stratum_mask])
        mapping = _assign_units_to_folds(stratum_clusters, n_folds, rng)
        stratum_idx = np.flatnonzero(stratum_mask)
        for i in stratum_idx:
            fold_ids[i] = mapping[cluster_keys[i]]
    return fold_ids


def _folds_stratified_only(
    strata: np.ndarray, n_folds: int, rng: np.random.Generator
) -> np.ndarray:
    """Assign folds for a stratified design with no clustering.

    Each stratum's rows are spread across all folds, so every fold contains
    observations from every stratum.

    Args:
        strata: Stratum labels.
        n_folds: Number of folds.
        rng: Seeded random generator.

    Returns:
        A 1-D integer array of fold indices, one per row.
    """
    fold_ids = np.full(len(strata), -1, dtype=int)
    for stratum in np.unique(strata):
        stratum_idx = np.flatnonzero(strata == stratum)
        shuffled = stratum_idx.copy()
        rng.shuffle(shuffled)
        fold_ids[shuffled] = np.arange(len(shuffled)) % n_folds
    return fold_ids


def design_aware_folds(
    strata: object = None,
    clusters: object = None,
    n_folds: int = 5,
    random_state: int | None = None,
) -> np.ndarray:
    """Construct cross-validation fold ids that respect a complex survey design.

    Implements the Wieczorek et al. (2022) rules so that clustered, stratified,
    and nested designs are handled correctly. The returned ids can be used to
    build a held-out test set or fed to :class:`surveycv.SurveyFold` for tuning.

    Args:
        strata: Array-like of stratum labels, or ``None`` for an unstratified
            design. Must be positionally aligned with the feature matrix.
        clusters: Array-like of cluster (PSU) labels, or ``None`` for an
            unclustered design. Must be positionally aligned with the features.
        n_folds: Number of folds to produce. Defaults to 5.
        random_state: Seed for reproducible fold assignment, or ``None`` for a
            fresh non-deterministic generator.

    Returns:
        A 1-D integer numpy array of length ``n_rows`` whose values lie in
        ``range(n_folds)``; entry ``i`` is the fold of observation ``i``.

    Raises:
        ValueError: If both ``strata`` and ``clusters`` are ``None``, if
            ``n_folds`` is invalid, or if the design arrays differ in length.

    Edge cases:
        Emits a warning (via the validation layer) when any stratum has fewer
        sampling units than ``n_folds``, since such strata cannot populate every
        fold. The folds are still returned so the caller can decide whether to
        reduce ``n_folds``.
    """
    check_n_folds(n_folds)

    strata_arr = to_1d_array(strata, "strata") if strata is not None else None
    clusters_arr = to_1d_array(clusters, "clusters") if clusters is not None else None
    require_design(strata_arr, clusters_arr)

    n_rows = len(strata_arr) if strata_arr is not None else len(clusters_arr)
    align_lengths(n_rows, strata_arr, clusters_arr)

    cluster_keys = (
        cluster_keys_within_strata(strata_arr, clusters_arr) if clusters_arr is not None else None
    )
    warn_if_infeasible(strata_arr, cluster_keys, n_folds)

    rng = np.random.default_rng(random_state)

    if cluster_keys is not None:
        return _folds_clustered(strata_arr, cluster_keys, n_folds, rng)
    return _folds_stratified_only(strata_arr, n_folds, rng)
