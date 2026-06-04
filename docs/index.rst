surveycv
========

Design-aware cross-validation and train/test splitting for complex sample
surveys, for scikit-learn / XGBoost / LightGBM pipelines in Python.

When data comes from a stratified, clustered survey (YRBS, NSFG, NHANES, BRFSS,
and similar), ordinary K-fold cross-validation leaks information: respondents
from the same primary sampling unit (PSU, for example a school) can land in both
the training and test folds, so the model memorizes cluster-specific patterns
and its measured performance is optimistic. ``surveycv`` builds folds and
train/test splits that keep each PSU wholly on one side of the split and stay
balanced across strata, and it scores test folds with survey weights so model
selection targets a population-representative objective.

``surveycv`` is a clean-room Python implementation of the methodology in
Wieczorek, Guerin and McMahon (2022), and the companion R package
`surveyCV <https://cran.r-project.org/package=surveyCV>`_. It contains no code
from that package and is released under the MIT license. See
:doc:`methodology` for the background and :doc:`references` for citations.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   quickstart
   methodology
   api
   references

What it gives you
-----------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Function / class
     - Purpose
   * - :func:`~surveycv.design_aware_folds`
     - Per-row integer fold ids that respect the design.
   * - :class:`~surveycv.SurveyFold`
     - A scikit-learn cross-validator; drops into ``GridSearchCV``,
       ``cross_val_score``, or an Optuna objective.
   * - :func:`~surveycv.survey_train_test_split`
     - A single held-out split by whole clusters, stratum-balanced.
   * - :func:`~surveycv.cross_val_score_survey`
     - Design-aware CV that scores each test fold with its survey weights.

Clusters are always treated as nested within strata, which is correct for any
properly nested survey design (a PSU belongs to exactly one stratum).

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
