"""Tests for the cluster bootstrap confidence interval."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from surveycv import ClusterBootstrapResult, cluster_bootstrap_ci


def _design(n_strata: int = 12, psus: int = 8, rows: int = 25, seed: int = 0):
    """Build a nested design plus an outcome and weights.

    Returns:
        Tuple ``(strata, clusters, y, w)`` of aligned 1-D arrays.
    """
    rng = np.random.default_rng(seed)
    strata, clusters, y, w = [], [], [], []
    for s in range(n_strata):
        for p in range(psus):
            school_effect = rng.normal(0, 1.0)
            for _ in range(rows):
                strata.append(s)
                clusters.append(p)
                prob = 1.0 / (1.0 + np.exp(-(school_effect - 1.0)))
                y.append(int(rng.random() < prob))
                w.append(rng.uniform(0.5, 3.0))
    return (np.array(strata), np.array(clusters), np.array(y), np.array(w, dtype=float))


def test_estimate_matches_statistic_on_full_sample():
    """The reported estimate equals the statistic evaluated on every row."""
    strata, clusters, y, w = _design()

    def weighted_prev(idx):
        return float(np.average(y[idx], weights=w[idx]))

    result = cluster_bootstrap_ci(
        weighted_prev, clusters=clusters, strata=strata, n_boot=200, random_state=0
    )
    assert isinstance(result, ClusterBootstrapResult)
    assert result.estimate == pytest.approx(weighted_prev(np.arange(len(y))))


def test_ci_brackets_estimate_and_is_ordered():
    """The interval contains the point estimate and ci_low <= ci_high."""
    strata, clusters, y, w = _design()

    def weighted_prev(idx):
        return float(np.average(y[idx], weights=w[idx]))

    r = cluster_bootstrap_ci(
        weighted_prev, clusters=clusters, strata=strata, n_boot=500, random_state=1
    )
    assert r.ci_low <= r.estimate <= r.ci_high
    assert r.ci_low < r.ci_high
    assert r.standard_error > 0


def test_determinism_with_seed():
    """Same seed yields identical bounds; a different seed differs."""
    strata, clusters, y, w = _design()
    stat = lambda idx: float(np.average(y[idx], weights=w[idx]))  # noqa: E731
    a = cluster_bootstrap_ci(stat, clusters=clusters, strata=strata, n_boot=300, random_state=7)
    b = cluster_bootstrap_ci(stat, clusters=clusters, strata=strata, n_boot=300, random_state=7)
    c = cluster_bootstrap_ci(stat, clusters=clusters, strata=strata, n_boot=300, random_state=8)
    assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)
    assert (a.ci_low, a.ci_high) != (c.ci_low, c.ci_high)


def test_resamples_contain_whole_psus_only():
    """Every resample is built from whole PSUs (row counts are sums of PSU sizes)."""
    strata, clusters, y, w = _design(n_strata=4, psus=5, rows=10)
    keys = np.array([f"{s}:{c}" for s, c in zip(strata, clusters)])
    psu_sizes = {k: int((keys == k).sum()) for k in np.unique(keys)}

    seen_sizes = []

    def record_size(idx):
        seen_sizes.append(len(idx))
        return float(len(idx))

    cluster_bootstrap_ci(record_size, clusters=clusters, strata=strata, n_boot=50, random_state=0)
    # Each resample size must be expressible as a sum of whole-PSU sizes; since
    # all PSUs here are the same size, every resample length is a multiple of it.
    unit = next(iter(set(psu_sizes.values())))
    assert all(size % unit == 0 for size in seen_sizes[1:])  # skip the full-sample call


def test_unstratified_design_resamples_clusters():
    """With strata=None the bootstrap resamples clusters across the whole sample."""
    _, clusters, y, w = _design()
    stat = lambda idx: float(np.average(y[idx], weights=w[idx]))  # noqa: E731
    r = cluster_bootstrap_ci(stat, clusters=clusters, n_boot=300, random_state=0)
    assert r.ci_low < r.estimate < r.ci_high


def test_single_psu_stratum_warns():
    """A stratum with one PSU triggers the lonely-PSU warning."""
    strata = np.array([0, 0, 0, 1, 1, 1])
    clusters = np.array([0, 1, 2, 0, 0, 0])  # stratum 1 has a single PSU
    y = np.array([0, 1, 0, 1, 0, 1])
    with pytest.warns(UserWarning, match="single PSU"):
        cluster_bootstrap_ci(
            lambda idx: float(y[idx].mean()),
            clusters=clusters,
            strata=strata,
            n_boot=50,
            random_state=0,
        )


def test_non_finite_resamples_excluded_with_warning():
    """Resamples whose statistic is nan are dropped and counted, with a warning."""
    strata, clusters, y, w = _design(n_strata=3, psus=4, rows=8)

    calls = {"n": 0}

    def flaky(idx):
        calls["n"] += 1
        # Return nan on a deterministic subset of resample calls.
        return float("nan") if calls["n"] % 3 == 0 else float(y[idx].mean())

    with pytest.warns(UserWarning, match="non-finite"):
        r = cluster_bootstrap_ci(flaky, clusters=clusters, strata=strata, n_boot=30, random_state=0)
    assert r.n_boot < 30


def test_all_non_finite_raises():
    """If every resample is non-finite, a clear error is raised."""
    strata, clusters, _, _ = _design(n_strata=3, psus=4, rows=8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="non-finite"):
            cluster_bootstrap_ci(
                lambda idx: float("nan"),
                clusters=clusters,
                strata=strata,
                n_boot=20,
                random_state=0,
            )


def test_requires_clusters():
    """Omitting clusters is an error."""
    with pytest.raises(ValueError, match="clusters must be provided"):
        cluster_bootstrap_ci(lambda idx: 0.0, clusters=None, strata=np.array([0, 1]))


def test_non_callable_statistic_raises():
    """A non-callable statistic is rejected."""
    _, clusters, _, _ = _design(n_strata=2, psus=3, rows=5)
    with pytest.raises(TypeError, match="callable"):
        cluster_bootstrap_ci(42, clusters=clusters)


def test_invalid_alpha_and_n_boot_raise():
    """Out-of-range alpha or n_boot is rejected."""
    _, clusters, _, _ = _design(n_strata=2, psus=3, rows=5)
    with pytest.raises(ValueError, match="alpha"):
        cluster_bootstrap_ci(lambda idx: 0.0, clusters=clusters, alpha=1.0)
    with pytest.raises(ValueError, match="n_boot"):
        cluster_bootstrap_ci(lambda idx: 0.0, clusters=clusters, n_boot=0)


def test_mismatched_lengths_raise():
    """Strata and clusters of differing length are rejected."""
    with pytest.raises(ValueError, match="length"):
        cluster_bootstrap_ci(lambda idx: 0.0, clusters=np.array([0, 1, 2]), strata=np.array([0, 1]))


def test_bootstrap_distribution_returned_and_consistent():
    """The raw resample values are returned and match the reported summary."""
    strata, clusters, y, w = _design()

    def weighted_prev(idx):
        return float(np.average(y[idx], weights=w[idx]))

    r = cluster_bootstrap_ci(
        weighted_prev, clusters=clusters, strata=strata, n_boot=500, random_state=3
    )
    dist = r.bootstrap_distribution
    assert isinstance(dist, np.ndarray)
    assert dist.shape == (r.n_boot,)
    assert np.all(np.isfinite(dist))
    # The reported CI bounds are the percentiles of the returned distribution.
    assert r.ci_low == pytest.approx(np.percentile(dist, 2.5))
    assert r.ci_high == pytest.approx(np.percentile(dist, 97.5))
    # And the standard error is the (ddof=1) std of the distribution.
    assert r.standard_error == pytest.approx(np.std(dist, ddof=1))


def test_distribution_excludes_non_finite_resamples():
    """Non-finite resamples are not present in the returned distribution."""
    strata, clusters, y, _ = _design(n_strata=3, psus=4, rows=8)
    calls = {"n": 0}

    def flaky(idx):
        calls["n"] += 1
        return float("nan") if calls["n"] % 4 == 0 else float(y[idx].mean())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = cluster_bootstrap_ci(flaky, clusters=clusters, strata=strata, n_boot=40, random_state=0)
    assert len(r.bootstrap_distribution) == r.n_boot
    assert np.all(np.isfinite(r.bootstrap_distribution))


def test_distribution_supports_bootstrap_pvalue():
    """The raw values enable a one-sample bootstrap p-value (share at/over 0.5)."""
    strata, clusters, y, w = _design()

    def weighted_prev(idx):
        return float(np.average(y[idx], weights=w[idx]))

    r = cluster_bootstrap_ci(
        weighted_prev, clusters=clusters, strata=strata, n_boot=500, random_state=1
    )
    # Prevalence is well below 0.5 here, so essentially no resample reaches it.
    share_ge_half = float(np.mean(r.bootstrap_distribution >= 0.5))
    assert 0.0 <= share_ge_half <= 1.0
    assert share_ge_half < 0.05
