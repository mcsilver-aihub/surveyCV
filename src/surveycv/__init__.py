"""surveycv: design-aware cross-validation for complex sample surveys.

A clean-room Python implementation of the fold-construction methodology from
Wieczorek, Guerin and McMahon (2022), "K-fold cross-validation for complex
sample surveys", Stat 11(1):e454, and the surveyCV R package. Folds and
train/test splits respect stratum and cluster (PSU) structure so that
prediction-error estimates and model selection are not biased by the sampling
design.

Public API:
    design_aware_folds: build per-row fold ids that respect the design.
    SurveyFold: a scikit-learn cross-validator wrapping those folds.
    survey_train_test_split: a single held-out split by whole clusters.
    cross_val_score_survey: design-aware, survey-weighted cross-validation.
    cluster_bootstrap_ci: confidence intervals by resampling PSUs within strata.
"""

from __future__ import annotations

from .bootstrap import ClusterBootstrapResult, cluster_bootstrap_ci
from .folds import design_aware_folds
from .scoring import cross_val_score_survey
from .split import survey_train_test_split
from .splitter import SurveyFold

__all__ = [
    "design_aware_folds",
    "SurveyFold",
    "survey_train_test_split",
    "cross_val_score_survey",
    "cluster_bootstrap_ci",
    "ClusterBootstrapResult",
]

__version__ = "0.2.0"
