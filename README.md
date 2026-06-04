# surveycv

Design-aware cross-validation and train/test splitting for complex sample
surveys, for scikit-learn / XGBoost / LightGBM pipelines in Python.

When data comes from a stratified, clustered survey (YRBS, NSFG, NHANES, BRFSS,
and similar), ordinary K-fold cross-validation leaks information: students from
the same school (the primary sampling unit, or PSU) can land in both the
training and test folds, so the model memorizes cluster-specific patterns and
its measured performance is optimistic. `surveycv` builds folds and train/test
splits that keep each PSU wholly on one side of the split and stay balanced
across strata, and it scores test folds with survey weights so model selection
targets a population-representative objective.

## Background and original research

The methodology comes from a peer-reviewed paper:

> **Wieczorek, J., Guerin, C., & McMahon, T. (2022). K-fold cross-validation for
> complex sample surveys.** *Stat*, 11(1), e454.
> https://doi.org/10.1002/sta4.454

That paper shows that when data is collected under a complex sampling design,
cross-validation folds must mirror the design or both the prediction-error
estimate and the model selection it drives become biased. The governing rules
are: for cluster (PSU) sampling, every observation from a cluster must go in the
same fold; for stratified sampling, folds must be balanced across strata; and
the held-out loss should be survey-weighted so it estimates a
population-representative error.

The authors also published a companion R package,
[surveyCV](https://cran.r-project.org/package=surveyCV) (Wieczorek, Guerin,
McMahon & Ratliff), whose worked example uses NSFG, a survey with the same
nested stratum-then-PSU structure as YRBS.

`surveycv` is a **clean-room Python implementation** of that methodology. It
contains no code from the R package, is released under the MIT license, and
exists so the method is usable directly inside scikit-learn / XGBoost / LightGBM
workflows without crossing into R. Please cite Wieczorek et al. (2022) when you
use it (see [Citation](#citation)).

## Install

```bash
pip install surveycv
```

Requires Python 3.9+, numpy, and scikit-learn.

## Documentation

Hosted documentation: **https://mcsilver-aihub.github.io/surveyCV/**

The Sphinx sources (API reference, methodology background, and worked examples)
live in [`docs/`](docs/). Every push to `main` rebuilds and publishes the site
via GitHub Pages. A [Read the Docs](https://readthedocs.org/) configuration
(`.readthedocs.yaml`) is also included. Build it locally with:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

## What it gives you

| Function / class | Purpose |
| --- | --- |
| `design_aware_folds(strata, clusters, n_folds, random_state)` | Per-row integer fold ids that respect the design. |
| `SurveyFold(n_splits, strata, clusters, random_state)` | A scikit-learn cross-validator; drops into `GridSearchCV`, `cross_val_score`, or an Optuna objective. |
| `survey_train_test_split(strata, clusters, test_size, random_state)` | A single held-out split by whole clusters, stratum-balanced. |
| `cross_val_score_survey(estimator, X, y, cv, weights, scoring)` | Design-aware CV that scores each test fold with its survey weights. |

Clusters are always treated as nested within strata, which is correct for any
properly nested survey design (a PSU belongs to exactly one stratum).

## Quick start

```python
import numpy as np
from surveycv import survey_train_test_split, SurveyFold, cross_val_score_survey
from xgboost import XGBClassifier

# strata, psu, weight come straight from the survey file (e.g. YRBS columns
# `stratum`, `PSU`, `weight`); X and y are your features and outcome.

# 1. Primary train/test split that never puts a school in both sides.
train_idx, test_idx = survey_train_test_split(
    strata=stratum, clusters=psu, test_size=0.2, random_state=42
)
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# 2. Design-aware, survey-weighted cross-validation for tuning.
cv = SurveyFold(
    n_splits=5,
    strata=stratum[train_idx],
    clusters=psu[train_idx],
    random_state=42,
)
scores = cross_val_score_survey(
    XGBClassifier(eval_metric="logloss"),
    X_train, y_train,
    cv=cv,
    weights=weight[train_idx],
    scoring="roc_auc",
)
print(scores.mean())
```

### Use as a scikit-learn `cv` object

`SurveyFold` follows the scikit-learn cross-validator API, so it works anywhere
a `cv` is accepted:

```python
from sklearn.model_selection import GridSearchCV

cv = SurveyFold(n_splits=5, strata=stratum, clusters=psu, random_state=0)
search = GridSearchCV(estimator, param_grid, cv=cv)
search.fit(X, y)
```

The stratum and cluster arrays are fixed at construction and must be
positionally aligned with the `X` you later pass to `split` / `fit`.

### Optuna objective

```python
import optuna
from surveycv import SurveyFold, cross_val_score_survey

cv = SurveyFold(n_splits=5, strata=stratum, clusters=psu, random_state=0)

def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
    }
    scores = cross_val_score_survey(
        XGBClassifier(**params, eval_metric="logloss"),
        X, y, cv=cv, weights=weight, scoring="roc_auc",
    )
    return scores.mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
```

## Fold-feasibility constraint

A design supports at most as many folds as the smallest stratum has PSUs. If you
request more folds than that, `surveycv` warns and tells you the largest
feasible fold count (the Wieczorek NSFG example was limited to 4-fold CV for
exactly this reason). Check your smallest stratum's PSU count before fixing
`n_folds`.

## Supported scoring names

`cross_val_score_survey` currently supports `roc_auc`, `accuracy`, and
`neg_mean_squared_error`, each computed with survey weights. Higher is always
better, matching scikit-learn's scorer convention.

## Citation

If you use `surveycv` in published work, please cite the original methodology
paper:

```bibtex
@article{wieczorek2022kfold,
  title   = {K-fold cross-validation for complex sample surveys},
  author  = {Wieczorek, Jerzy and Guerin, Cole and McMahon, Thomas},
  journal = {Stat},
  volume  = {11},
  number  = {1},
  pages   = {e454},
  year    = {2022},
  doi     = {10.1002/sta4.454}
}
```

## References

- Wieczorek, J., Guerin, C., & McMahon, T. (2022). K-fold cross-validation for
  complex sample surveys. *Stat*, 11(1), e454.
  https://doi.org/10.1002/sta4.454
- surveyCV R package (Wieczorek, Guerin, McMahon & Ratliff). CRAN.
  https://cran.r-project.org/package=surveyCV
- Wolter, K. M. (2007). *Introduction to Variance Estimation* (2nd ed.).
  Springer. (Random groups, the group jackknife, and the cluster bootstrap for
  clustered survey data.)
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python.
  *Journal of Machine Learning Research*, 12, 2825-2830. (Cross-validator API.)

## License

MIT. The methodology is due to Wieczorek et al. (2022); please cite it (see
[Citation](#citation)). This package is an independent clean-room
implementation and is not affiliated with or endorsed by the original authors.
