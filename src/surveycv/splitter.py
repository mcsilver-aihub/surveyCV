"""A scikit-learn cross-validator that honors a complex survey design.

:class:`SurveyFold` wraps :func:`surveycv.design_aware_folds` so design-aware
folds can be passed anywhere scikit-learn expects a ``cv`` object: tuners such
as ``GridSearchCV`` and ``RandomizedSearchCV``, ``cross_val_score``, or an
Optuna objective that calls ``cross_val_score`` internally.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from .folds import design_aware_folds


class SurveyFold:
    """Design-aware K-fold splitter compatible with the scikit-learn CV API.

    The stratum and cluster labels are supplied at construction time and must be
    positionally aligned with the ``X`` later passed to :meth:`split`. This
    mirrors ``sklearn.model_selection.PredefinedSplit``, which also fixes the
    partition before seeing ``X``.

    Args:
        n_splits: Number of folds. Defaults to 5.
        strata: Array-like of stratum labels, or ``None`` for an unstratified
            design.
        clusters: Array-like of cluster (PSU) labels, or ``None`` for an
            unclustered design. Clusters are treated as nested within strata.
        random_state: Seed for reproducible folds, or ``None``.

    Raises:
        ValueError: Propagated from :func:`design_aware_folds` for invalid
            designs or fold counts.

    Edge cases:
        Folds are computed lazily on the first :meth:`split` call and cached, so
        repeated splits with the same instance are deterministic regardless of
        ``random_state`` being a one-shot seed.
    """

    def __init__(
        self,
        n_splits: int = 5,
        strata: object = None,
        clusters: object = None,
        random_state: int | None = None,
    ) -> None:
        """Store the design labels and fold settings; see the class docstring."""
        self.n_splits = n_splits
        self.strata = strata
        self.clusters = clusters
        self.random_state = random_state
        self._fold_ids: np.ndarray | None = None

    def _compute_fold_ids(self, n_rows: int) -> np.ndarray:
        """Compute and cache the per-row fold ids for ``n_rows`` observations.

        Args:
            n_rows: Number of rows in the feature matrix passed to ``split``.

        Returns:
            The cached 1-D array of fold ids.

        Raises:
            ValueError: If ``n_rows`` does not match the length of the design
                labels supplied at construction.
        """
        if self._fold_ids is None:
            self._fold_ids = design_aware_folds(
                strata=self.strata,
                clusters=self.clusters,
                n_folds=self.n_splits,
                random_state=self.random_state,
            )
        if len(self._fold_ids) != n_rows:
            raise ValueError(
                f"X has {n_rows} rows but the design labels describe "
                f"{len(self._fold_ids)} rows; they must be aligned."
            )
        return self._fold_ids

    def split(
        self,
        X: object,
        y: object = None,
        groups: object = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield train/test index arrays for each design-aware fold.

        Args:
            X: Feature matrix (or any sized container) whose row count and order
                match the design labels given at construction.
            y: Ignored; present for scikit-learn API compatibility.
            groups: Ignored; the grouping comes from the design labels, not from
                this argument.

        Yields:
            Tuples ``(train_index, test_index)`` of integer numpy arrays, with
            fold ``k`` held out as the test set in turn.

        Raises:
            ValueError: If ``X`` does not align with the design labels.
        """
        n_rows = _n_rows(X)
        fold_ids = self._compute_fold_ids(n_rows)
        all_index = np.arange(n_rows)
        for fold in range(self.n_splits):
            test_mask = fold_ids == fold
            yield all_index[~test_mask], all_index[test_mask]

    def get_n_splits(self, X: object = None, y: object = None, groups: object = None) -> int:
        """Return the number of folds, per the scikit-learn cross-validator API.

        Args:
            X: Ignored.
            y: Ignored.
            groups: Ignored.

        Returns:
            The configured number of splits.
        """
        return self.n_splits


def _n_rows(X: object) -> int:
    """Return the row count of a feature container.

    Args:
        X: A numpy array, pandas DataFrame, or any object supporting ``len``.

    Returns:
        The number of rows.

    Raises:
        TypeError: If the row count cannot be determined from ``X``.
    """
    if hasattr(X, "shape"):
        return int(X.shape[0])
    try:
        return len(X)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(
            "Cannot determine the number of rows in X; pass a numpy array, "
            "pandas DataFrame, or a sized sequence."
        ) from exc
