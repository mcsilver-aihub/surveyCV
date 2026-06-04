Methodology and Original Research
=================================

``surveycv`` is a clean-room implementation of the cross-validation methodology
for complex sample surveys developed in:

   Wieczorek, J., Guerin, C., & McMahon, T. (2022).
   **K-fold cross-validation for complex sample surveys.**
   *Stat*, 11(1), e454. https://doi.org/10.1002/sta4.454

This page summarizes the problem that paper solves, the fold-assignment rules it
prescribes, and how this package realizes them. It is a description of the
published method, not a substitute for the paper; please read and cite the
original (see :doc:`references`).

The problem
-----------

Most machine-learning tooling assumes the rows of a dataset are independent and
identically distributed. Data from a complex survey is not: it is collected
under a sampling design that typically combines

- **stratification** -- the population is divided into strata (e.g. region or
  school type) and each stratum is sampled separately, and
- **clustering** -- within a stratum, whole primary sampling units (PSUs, e.g.
  schools) are drawn, and then respondents within each selected PSU, and
- **unequal selection probabilities** -- recorded as survey weights so that
  estimates can be made representative of the target population.

Ordinary K-fold cross-validation ignores all three. When it splits rows at
random, respondents from the same PSU land in different folds. A flexible model
can then learn cluster-specific signal in the training fold and be rewarded for
it in the test fold, because the same clusters appear on both sides. The
resulting cross-validation error is optimistic, and because hyperparameters are
chosen to minimize that error, model selection is biased toward models that
overfit the clusters.

The paper's solution
--------------------

Wieczorek et al. (2022) show that the fix is to make the folds mirror the
sampling design. The fold-assignment rules are:

#. **Cluster (PSU) sampling.** Every observation belonging to a given PSU must be
   placed in the same fold. A PSU is never split across folds.
#. **Stratified sampling.** Folds must be balanced across strata: each fold
   should draw observations (or PSUs) from every stratum, so no fold is
   dominated by a subset of the population.
#. **Nested designs (clusters within strata).** For surveys such as NSFG and
   YRBS, where PSUs are nested in strata, each stratum's PSUs are partitioned
   across the folds independently.
#. **Survey-weighted test error.** The held-out loss is averaged using the
   survey weights (a Horvitz-Thompson style weighting), so the cross-validation
   objective estimates a population-representative error rather than the
   unweighted sample mean.

A direct consequence is a **feasibility constraint**: a design supports at most
as many folds as the smallest stratum has PSUs. The paper's NSFG example is
limited to 4-fold cross-validation because some strata contain only four PSUs.

How ``surveycv`` realizes the method
------------------------------------

The package maps each rule onto a small, composable piece of API.

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Paper concept
     - ``surveycv`` API
   * - PSU-intact, stratum-balanced fold ids
     - :func:`~surveycv.design_aware_folds`
   * - Those folds as a scikit-learn cross-validator
     - :class:`~surveycv.SurveyFold`
   * - A single design-correct held-out evaluation split
     - :func:`~surveycv.survey_train_test_split`
   * - Survey-weighted cross-validation scoring
     - :func:`~surveycv.cross_val_score_survey`

Fold construction partitions the unique PSUs within each stratum across the
folds, then assigns every row to its PSU's fold, which enforces rules 1-3
simultaneously. Scoring passes the held-out rows' survey weights to the metric,
which enforces rule 4. The feasibility constraint is surfaced as a warning when a
stratum has fewer PSUs than the requested fold count.

Relationship to the surveyCV R package
--------------------------------------

The original authors released an R package,
`surveyCV <https://cran.r-project.org/package=surveyCV>`_, that implements the
same methodology and whose worked example uses NSFG, a survey with the same
nested stratum-then-PSU structure as YRBS. ``surveycv`` (this package) is an
independent Python implementation written from the published method. It copies
no code from the R package and is not affiliated with or endorsed by its
authors; it exists so the method can be used directly inside Python
scikit-learn / XGBoost / LightGBM workflows.

A note on ``nest``
------------------

The R package exposes a ``nest`` argument to indicate that PSU labels are reused
across strata (as in YRBS, where PSU ``1`` appears in every stratum). ``surveycv``
does not expose this argument because it always treats the ``(stratum, PSU)``
pair as the sampling unit. That is correct for any properly nested design, since
a real PSU belongs to exactly one stratum, so an already-unique PSU label is
unchanged by the pairing.
