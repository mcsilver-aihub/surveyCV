"""Shared input validation for design-aware cross-validation.

Lower-level helpers that raise specific exceptions on malformed input so the
public functions can stay focused on the fold-construction logic.
"""

from __future__ import annotations

import warnings

import numpy as np

MIN_FOLDS = 2


def to_1d_array(values: object, name: str) -> np.ndarray:
    """Coerce a sequence of design labels into a 1-D numpy array.

    Args:
        values: Any array-like of labels (strata or cluster ids). May be a
            list, pandas Series, or numpy array.
        name: Human-readable name of the argument, used in error messages.

    Returns:
        A 1-D numpy array view of ``values``.

    Raises:
        ValueError: If ``values`` is not one-dimensional.

    Edge cases:
        A pandas Series is converted via ``np.asarray``; its index is dropped,
        so callers must ensure positional alignment with the feature matrix.
    """
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional, got shape {array.shape}.")
    return array


def check_n_folds(n_folds: int) -> None:
    """Validate that the requested number of folds is usable.

    Args:
        n_folds: Number of cross-validation folds requested.

    Raises:
        ValueError: If ``n_folds`` is not an integer of at least ``MIN_FOLDS``.

    Edge cases:
        Booleans are rejected even though ``bool`` is a subclass of ``int``,
        since ``True``/``False`` are never a meaningful fold count.
    """
    if isinstance(n_folds, bool) or not isinstance(n_folds, (int, np.integer)):
        raise ValueError(f"n_folds must be an integer, got {type(n_folds).__name__}.")
    if n_folds < MIN_FOLDS:
        raise ValueError(f"n_folds must be at least {MIN_FOLDS}, got {n_folds}.")


def align_lengths(
    n_rows: int,
    strata: np.ndarray | None,
    clusters: np.ndarray | None,
    weights: np.ndarray | None = None,
) -> None:
    """Verify that every supplied design array matches the row count.

    Args:
        n_rows: Expected number of observations.
        strata: Stratum labels, or ``None`` if the design is unstratified.
        clusters: Cluster (PSU) labels, or ``None`` if there is no clustering.
        weights: Survey weights, or ``None`` if unweighted.

    Raises:
        ValueError: If any non-None array has a length other than ``n_rows``.
    """
    for array, name in ((strata, "strata"), (clusters, "clusters"), (weights, "weights")):
        if array is not None and len(array) != n_rows:
            raise ValueError(f"{name} has length {len(array)} but {n_rows} rows were expected.")


def require_design(strata: np.ndarray | None, clusters: np.ndarray | None) -> None:
    """Ensure at least one design dimension was provided.

    Args:
        strata: Stratum labels or ``None``.
        clusters: Cluster (PSU) labels or ``None``.

    Raises:
        ValueError: If both ``strata`` and ``clusters`` are ``None``.
    """
    if strata is None and clusters is None:
        raise ValueError(
            "At least one of strata or clusters must be provided; for simple "
            "random sampling use scikit-learn's KFold instead."
        )


def cluster_keys_within_strata(
    strata: np.ndarray | None,
    clusters: np.ndarray | None,
) -> np.ndarray:
    """Build per-row cluster keys that identify each PSU within its stratum.

    Survey datasets such as YRBS reuse PSU labels across strata (PSU ``1``
    appears in every stratum), so the true sampling unit is the
    ``(stratum, cluster)`` pair. When strata are present the cluster identity is
    therefore that compound pair; same-numbered PSUs in different strata become
    distinct units. This is always correct for a nested design because a PSU
    belongs to exactly one stratum, so a globally unique label is unchanged by
    the compounding.

    Args:
        strata: Stratum labels, or ``None`` if unstratified.
        clusters: Cluster (PSU) labels, or ``None`` if unclustered.

    Returns:
        A 1-D array of per-row cluster keys (strings when strata are present).

    Raises:
        ValueError: If ``clusters`` is ``None`` (there is nothing to key).

    Edge cases:
        When ``strata`` is ``None`` the cluster labels are returned unchanged,
        since there is no stratum to nest within.
    """
    if clusters is None:
        raise ValueError("clusters must be provided to build cluster keys.")
    if strata is not None:
        return np.array([f"{stratum}::{cluster}" for stratum, cluster in zip(strata, clusters)])
    return clusters


def warn_if_infeasible(
    strata: np.ndarray | None,
    cluster_keys: np.ndarray | None,
    n_folds: int,
) -> tuple[int, int]:
    """Warn when a stratum has fewer sampling units than requested folds.

    Wieczorek et al. note that a design supports at most as many folds as the
    smallest stratum has sampling units; their NSFG example was limited to
    4-fold cross-validation for this reason. This surfaces the same constraint
    rather than silently leaving folds empty for small strata.

    Args:
        strata: Stratum labels, or ``None`` if unstratified.
        cluster_keys: Per-row cluster keys, or ``None`` if unclustered (in which
            case rows themselves are the sampling units).
        n_folds: Requested number of folds.

    Returns:
        A tuple ``(min_units, n_small_strata)`` giving the smallest per-stratum
        sampling-unit count and how many strata fall below ``n_folds``.

    Edge cases:
        When ``strata`` is ``None`` the whole sample is one group, so the
        sampling-unit count is taken across all rows.
    """
    if strata is None:
        units = cluster_keys if cluster_keys is not None else np.arange(0)
        min_units = len(np.unique(units)) if cluster_keys is not None else n_folds
        return min_units, 0

    unique_strata = np.unique(strata)
    per_stratum_units = []
    for stratum in unique_strata:
        mask = strata == stratum
        if cluster_keys is not None:
            per_stratum_units.append(len(np.unique(cluster_keys[mask])))
        else:
            per_stratum_units.append(int(mask.sum()))

    per_stratum_units_arr = np.array(per_stratum_units)
    n_small = int((per_stratum_units_arr < n_folds).sum())
    min_units = int(per_stratum_units_arr.min())
    if n_small > 0:
        warnings.warn(
            f"{n_small} stratum/strata have fewer than n_folds={n_folds} sampling "
            f"units (smallest has {min_units}). Those strata cannot contribute to "
            f"every fold; consider reducing n_folds to {min_units}.",
            stacklevel=2,
        )
    return min_units, n_small
