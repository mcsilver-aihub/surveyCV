"""Cluster bootstrap confidence intervals for complex sample surveys.

Resamples whole primary sampling units (PSUs) with replacement, within strata,
to build a sampling distribution for an arbitrary survey-weighted statistic
(weighted prevalence, weighted AUC, sensitivity, and so on). This is the
design-correct way to put a confidence interval on a statistic computed from
clustered survey data.

It complements the cross-validation tools in this package: folds *partition* the
data for model selection, while the bootstrap *resamples* it (with replacement)
for variance estimation. The two are not interchangeable.

Reference for the cluster bootstrap: Wolter, K. M. (2007). Introduction to
Variance Estimation (2nd ed.), Springer.
"""

from __future__ import annotations

import warnings
from typing import Callable, NamedTuple

import numpy as np

from ._validation import (
    align_lengths,
    cluster_keys_within_strata,
    require_design,
    to_1d_array,
)

DEFAULT_N_BOOT = 2000
DEFAULT_ALPHA = 0.05
_PERCENT = 100.0
_MIN_N_BOOT = 1


class ClusterBootstrapResult(NamedTuple):
    """Outcome of a cluster bootstrap confidence interval.

    Attributes:
        estimate: The statistic evaluated once on the full sample.
        ci_low: Lower percentile bound of the bootstrap distribution.
        ci_high: Upper percentile bound of the bootstrap distribution.
        standard_error: Standard deviation of the bootstrap estimates.
        n_boot: Number of bootstrap resamples used (excludes any that produced a
            non-finite statistic).
        alpha: Two-sided significance level used for the interval.
        bootstrap_distribution: The 1-D array of finite resample statistics
            (length ``n_boot``) that the interval was read from. Useful for
            plotting the distribution, computing a bootstrap p-value, or building
            a different interval type without re-running the resampling.
    """

    estimate: float
    ci_low: float
    ci_high: float
    standard_error: float
    n_boot: int
    alpha: float
    bootstrap_distribution: np.ndarray


def _validate_alpha(alpha: float) -> None:
    """Validate the two-sided significance level.

    Args:
        alpha: Requested significance level.

    Raises:
        ValueError: If ``alpha`` is not strictly between 0 and 1.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be strictly between 0 and 1, got {alpha}.")


def _validate_n_boot(n_boot: int) -> None:
    """Validate the requested number of bootstrap resamples.

    Args:
        n_boot: Number of resamples.

    Raises:
        ValueError: If ``n_boot`` is not an integer of at least ``_MIN_N_BOOT``.

    Edge cases:
        Booleans are rejected even though ``bool`` subclasses ``int``.
    """
    if isinstance(n_boot, bool) or not isinstance(n_boot, (int, np.integer)):
        raise ValueError(f"n_boot must be an integer, got {type(n_boot).__name__}.")
    if n_boot < _MIN_N_BOOT:
        raise ValueError(f"n_boot must be at least {_MIN_N_BOOT}, got {n_boot}.")


def _group_psus(
    strata: np.ndarray | None, cluster_keys: np.ndarray, n_rows: int
) -> tuple[list[np.ndarray], dict]:
    """Index PSUs by stratum and map each PSU to its row positions.

    Args:
        strata: Stratum labels, or ``None`` for an unstratified design.
        cluster_keys: Per-row PSU keys (nested within strata when stratified).
        n_rows: Number of observations.

    Returns:
        A tuple ``(groups, psu_to_rows)`` where ``groups`` is a list of arrays of
        unique PSU keys (one array per stratum, or a single array for the whole
        sample when unstratified) and ``psu_to_rows`` maps each PSU key to its
        row indices.
    """
    rows = np.arange(n_rows)
    unique_keys = np.unique(cluster_keys)
    psu_to_rows = {key: rows[cluster_keys == key] for key in unique_keys}
    if strata is None:
        return [unique_keys], psu_to_rows
    groups = [np.unique(cluster_keys[strata == stratum]) for stratum in np.unique(strata)]
    return groups, psu_to_rows


def _resample_rows(
    groups: list[np.ndarray], psu_to_rows: dict, rng: np.random.Generator
) -> np.ndarray:
    """Draw one cluster-bootstrap resample of row indices.

    Within each stratum, PSUs are drawn with replacement to the same count the
    stratum originally had, and all rows of each drawn PSU are included.

    Args:
        groups: Per-stratum arrays of PSU keys.
        psu_to_rows: Mapping from PSU key to row indices.
        rng: Seeded random generator.

    Returns:
        A 1-D array of resampled row indices (with repeats).
    """
    drawn = []
    for keys in groups:
        sampled = rng.choice(keys, size=len(keys), replace=True)
        drawn.extend(psu_to_rows[key] for key in sampled)
    return np.concatenate(drawn)


def cluster_bootstrap_ci(
    statistic: Callable[[np.ndarray], float],
    *,
    clusters: object,
    strata: object = None,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = DEFAULT_ALPHA,
    random_state: int | None = None,
) -> ClusterBootstrapResult:
    """Percentile confidence interval by resampling whole PSUs within strata.

    Builds a sampling distribution for ``statistic`` by repeatedly resampling
    primary sampling units (PSUs) with replacement, independently within each
    stratum, and recomputing the statistic on each resample. This is the
    design-correct bootstrap for clustered survey data and is the right tool for
    confidence intervals (use :func:`design_aware_folds` / :class:`SurveyFold`
    for cross-validation, not for CIs).

    Args:
        statistic: A callable ``statistic(row_indices) -> float`` that computes
            the quantity of interest on the rows selected by the given positional
            indices (for example a survey-weighted AUC or weighted prevalence).
            It is called once on all rows for the point estimate and ``n_boot``
            times on resamples. It should return ``nan`` for a degenerate
            resample (such as one with a single outcome class) rather than raise.
        clusters: Array-like of cluster (PSU) labels, one per row. Required.
        strata: Array-like of stratum labels, or ``None`` to resample clusters
            across the whole sample. PSU labels are treated as nested within
            strata when ``strata`` is given.
        n_boot: Number of bootstrap resamples. Defaults to 2000.
        alpha: Two-sided significance level; the interval spans the
            ``alpha/2`` and ``1 - alpha/2`` percentiles. Defaults to 0.05.
        random_state: Seed for reproducible resampling, or ``None``.

    Returns:
        A :class:`ClusterBootstrapResult` with the point estimate, percentile
        confidence bounds, bootstrap standard error, the number of finite
        resamples used, ``alpha``, and ``bootstrap_distribution`` (the array of
        finite resample statistics, for plotting, a bootstrap p-value, or a
        custom interval).

    Raises:
        TypeError: If ``statistic`` is not callable.
        ValueError: If ``clusters`` is ``None``, the arrays disagree in length,
            ``n_boot`` or ``alpha`` is out of range, or every resample produced a
            non-finite statistic.

    Edge cases:
        Strata containing a single PSU contribute no within-stratum variance and
        trigger a warning. Resamples whose statistic is non-finite are dropped
        from the percentile computation (with a warning) so a few degenerate
        draws on a rare outcome do not invalidate the whole interval.

    Note:
        To compare two groups (for example a Black-vs-White sensitivity gap),
        make ``statistic`` return the difference computed on the *same* resample,
        e.g. ``lambda idx: metric(group_a in idx) - metric(group_b in idx)``, and
        read significance from whether the resulting interval excludes 0. Do not
        bootstrap each group separately and subtract the two distributions: when
        groups share PSUs (as race groups do within a school) the two group
        statistics are correlated, and subtracting independent draws ignores that
        covariance. With the positive correlation that shared schools induce, the
        gap interval then comes out too wide, so a real difference can be missed.
    """
    if not callable(statistic):
        raise TypeError("statistic must be callable: statistic(row_indices) -> float.")
    _validate_n_boot(n_boot)
    _validate_alpha(alpha)

    if clusters is None:
        raise ValueError("clusters must be provided for a cluster bootstrap.")
    strata_arr = to_1d_array(strata, "strata") if strata is not None else None
    clusters_arr = to_1d_array(clusters, "clusters")
    require_design(strata_arr, clusters_arr)

    n_rows = len(clusters_arr)
    align_lengths(n_rows, strata_arr, clusters_arr)
    cluster_keys = cluster_keys_within_strata(strata_arr, clusters_arr)
    groups, psu_to_rows = _group_psus(strata_arr, cluster_keys, n_rows)

    n_singleton_strata = sum(1 for keys in groups if len(keys) == 1)
    if strata_arr is not None and n_singleton_strata > 0:
        warnings.warn(
            f"{n_singleton_strata} stratum/strata contain a single PSU; they "
            "contribute no within-stratum variance to the bootstrap.",
            stacklevel=2,
        )

    estimate = float(statistic(np.arange(n_rows)))
    rng = np.random.default_rng(random_state)
    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        boot[b] = float(statistic(_resample_rows(groups, psu_to_rows, rng)))

    finite = np.isfinite(boot)
    n_finite = int(finite.sum())
    if n_finite < n_boot:
        warnings.warn(
            f"{n_boot - n_finite} of {n_boot} bootstrap resamples produced a "
            "non-finite statistic (e.g. a degenerate resample) and were excluded.",
            stacklevel=2,
        )
    if n_finite == 0:
        raise ValueError(
            "All bootstrap resamples produced a non-finite statistic; cannot form "
            "a confidence interval."
        )

    ci_low = float(np.nanpercentile(boot, _PERCENT * alpha / 2.0))
    ci_high = float(np.nanpercentile(boot, _PERCENT * (1.0 - alpha / 2.0)))
    standard_error = float(np.nanstd(boot, ddof=1)) if n_finite > 1 else float("nan")
    return ClusterBootstrapResult(
        estimate=estimate,
        ci_low=ci_low,
        ci_high=ci_high,
        standard_error=standard_error,
        n_boot=n_finite,
        alpha=alpha,
        bootstrap_distribution=boot[finite],
    )
