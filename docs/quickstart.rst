Quick Start
===========

The three inputs that drive every function are the survey design columns:

- ``strata`` -- the stratum label of each row (YRBS column ``stratum``)
- ``clusters`` -- the primary sampling unit / PSU of each row (YRBS column ``PSU``)
- ``weights`` -- the survey weight of each row (YRBS column ``weight``)

They must be positionally aligned with your feature matrix ``X`` and target ``y``.

Primary train/test split
-------------------------

Hold out whole clusters so no PSU (for example, no school) appears in both the
training and test sets:

.. code-block:: python

   from surveycv import survey_train_test_split

   train_idx, test_idx = survey_train_test_split(
       strata=stratum, clusters=psu, test_size=0.2, random_state=42
   )
   X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
   y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

Design-aware, survey-weighted cross-validation
----------------------------------------------

Build folds that respect the design and score each fold with its survey weights:

.. code-block:: python

   from surveycv import SurveyFold, cross_val_score_survey
   from xgboost import XGBClassifier

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

Use as a scikit-learn ``cv`` object
-----------------------------------

:class:`~surveycv.SurveyFold` follows the scikit-learn cross-validator API, so it
works anywhere a ``cv`` is accepted:

.. code-block:: python

   from sklearn.model_selection import GridSearchCV

   cv = SurveyFold(n_splits=5, strata=stratum, clusters=psu, random_state=0)
   search = GridSearchCV(estimator, param_grid, cv=cv)
   search.fit(X, y)

The stratum and cluster arrays are fixed at construction and must be
positionally aligned with the ``X`` you later pass to ``split`` / ``fit``.

Optuna objective
----------------

.. code-block:: python

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

Fold-feasibility constraint
---------------------------

A design supports at most as many folds as the smallest stratum has PSUs. If you
request more folds than that, ``surveycv`` warns and reports the largest feasible
fold count. The Wieczorek NSFG example was limited to 4-fold cross-validation for
exactly this reason, since some strata had only four PSUs. Check your smallest
stratum's PSU count before fixing ``n_folds``.

Supported scoring names
-----------------------

:func:`~surveycv.cross_val_score_survey` supports ``roc_auc``, ``accuracy``, and
``neg_mean_squared_error``, each computed with survey weights. Higher is always
better, matching scikit-learn's scorer convention.

Confidence intervals with the cluster bootstrap
-----------------------------------------------

For a confidence interval on a survey statistic (weighted prevalence, weighted
AUC, sensitivity, ...), resample whole PSUs with replacement within strata. Pass
a ``statistic(row_indices) -> float`` that computes your quantity on the rows
selected by the given positional indices:

.. code-block:: python

   import numpy as np
   from sklearn.metrics import roc_auc_score
   from surveycv import cluster_bootstrap_ci

   # y, proba, weight, stratum, psu are arrays aligned to the evaluation set.
   def weighted_auc(idx):
       return roc_auc_score(y[idx], proba[idx], sample_weight=weight[idx])

   result = cluster_bootstrap_ci(
       weighted_auc, clusters=psu, strata=stratum, n_boot=2000, random_state=0,
   )
   print(f"AUC {result.estimate:.3f} "
         f"(95% CI {result.ci_low:.3f}-{result.ci_high:.3f})")

The statistic should return ``nan`` for a degenerate resample (for example one
with a single outcome class on a rare outcome); such resamples are dropped from
the percentile computation and the surviving count is reported in
``result.n_boot``. Use the cluster bootstrap for confidence intervals, not the
cross-validation folds.
