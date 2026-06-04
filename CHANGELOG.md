# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
semantic versioning.

## [0.1.0] - 2026-06-04

### Added
- `design_aware_folds`: per-row fold ids that respect stratum and cluster (PSU)
  structure, implementing the Wieczorek et al. (2022) fold-assignment rules.
- `SurveyFold`: a scikit-learn-compatible cross-validator wrapping
  `design_aware_folds` for use with `GridSearchCV`, `cross_val_score`, and
  Optuna.
- `survey_train_test_split`: a single held-out split that assigns whole clusters
  to train or test while keeping every stratum represented.
- `cross_val_score_survey`: design-aware cross-validation that scores each test
  fold with its survey weights, supporting `roc_auc`, `accuracy`, and
  `neg_mean_squared_error`.
- Feasibility warning when a stratum has fewer PSUs than the requested fold
  count.
