# surveycv-py — Python port of surveyCV (design-aware CV for complex surveys)

Clean-room reimplementation of the fold-construction methodology from
Wieczorek, Guerin, McMahon (2022), "K-fold cross-validation for complex sample
surveys", Stat 11(1):e454. Base reference: surveyCV 0.2.0 (CRAN, GPL).
This port is original code under MIT; it cites the paper/package as the method
basis but copies no GPL source.

## Goal
A standalone, pip-installable module that gives a Python/scikit-learn pipeline
the same design-aware cross-validation surveyCV gives R: folds that respect
PSU (cluster) and stratum structure, plus survey-weighted scoring. Built to be
embedded into research_pipeline afterward.

## Todo

### Package scaffolding
- [ ] pyproject.toml (PEP 621, src layout, MIT, requires-python >=3.9)
- [ ] LICENSE (MIT)
- [ ] README.md (usage, methodology citation, YRBS-style example)
- [ ] CHANGELOG.md, .gitignore

### Core code (src/surveycv/)
- [ ] folds.py — `design_aware_folds(strata, clusters, n_folds, nest, shuffle, random_state)`
      Cluster rule: all rows of a PSU in one fold. Stratified: folds balanced
      across strata. Nested: partition each stratum's PSUs into folds. SRS: random.
- [ ] splitter.py — `SurveyFold` (sklearn BaseCrossValidator) wrapping design_aware_folds
      so it drops into GridSearchCV / Optuna / cross_val_score.
- [ ] split.py — `survey_train_test_split(...)` holding out a fraction of PSUs
      per stratum (the headline train/test split Brittney needs).
- [ ] scoring.py — survey-weighted metric wrappers + `cross_val_score_survey(...)`
      CV driver that fits per fold and scores with the correct weight slice
      (Horvitz-Thompson weighting), avoiding sklearn sample_weight routing issues.
- [ ] _validation.py — shared input validation (lengths, dtypes, fold feasibility).
- [ ] __init__.py — public API + __version__.

### Edge cases to handle explicitly
- [ ] Stratum with fewer PSUs than n_folds (the 4-fold-vs-5-fold YRBS constraint
      from the vignette): warn, do not silently misassign.
- [ ] nest=True relabels PSU ids within strata (YRBS reuses PSU labels per stratum).
- [ ] Cluster present but no strata; strata present but no clusters; neither.
- [ ] Reproducibility via random_state.

### Tests (tests/)
- [ ] test_folds.py — PSU integrity (no PSU split across folds), stratum balance,
      nesting, determinism, feasibility warnings.
- [ ] test_splitter.py — sklearn API compliance, no train/test PSU leakage.
- [ ] test_split.py — per-stratum holdout, no PSU leakage, ratio correctness.
- [ ] test_scoring.py — weighted metrics match manual HT computation.

### Finish
- [ ] ruff check + format clean
- [ ] pytest green
- [ ] Build sdist+wheel locally (python -m build), verify install in temp venv
- [ ] Confirm PyPI distribution name availability before any upload (separate step)
- [ ] Review section in this file

## Open follow-up (after module works)
- Embed the core (folds + weighted scorer) into research_pipeline for the
  YRBS PSU-stratified rebuild, and wire the survey-weighted scorer into Optuna.

## Review

### What was built (v0.1.0)
Standalone, pip-installable `surveycv` package under `~/AI_Hub/surveycv-py`,
clean-room from the Wieczorek et al. (2022) algorithm (no GPL source copied),
MIT licensed.

Public API (`src/surveycv/`):
- `design_aware_folds` (folds.py) — per-row fold ids; PSUs kept intact, folds
  balanced across strata, nested handling for YRBS/NSFG designs.
- `SurveyFold` (splitter.py) — scikit-learn cross-validator; works in
  GridSearchCV / cross_val_score / Optuna.
- `survey_train_test_split` (split.py) — single held-out split by whole
  clusters, every stratum represented.
- `cross_val_score_survey` (scoring.py) — design-aware CV with survey-weighted
  test-fold scoring (roc_auc / accuracy / neg_mean_squared_error).
- `_validation.py` — shared input checks + the fold-feasibility warning.

### Design decision worth noting
Dropped the `nest` parameter that surveyCV exposes. Because every code path
partitions PSUs within each stratum, and a real PSU belongs to exactly one
stratum, the `(stratum, PSU)` pair is always the correct unit and `nest` was
inert for valid input. Removing it keeps the contract honest; clusters are
always nested within strata.

### Verification
- 32 pytest tests pass (fold integrity, no PSU leakage, stratum balance,
  determinism, weighted-score correctness vs manual HT, sklearn integration,
  pandas support, error/warn paths).
- `ruff check` and `ruff format` clean.
- `python -m build` produces sdist + wheel; `twine check` PASSED on both.
- Wheel installs in a clean venv and runs end to end on a YRBS-shaped design
  (17 strata x 10 PSUs x 30 rows): no PSU leakage, balanced 5-fold split.
- PyPI name `surveycv` confirmed available (404).

### Not yet done (deliberately deferred)
- Actual PyPI upload (needs your account / API token; `twine upload` is the
  final step once you approve the name).
- Embedding the core into research_pipeline for the YRBS PSU rebuild (the
  agreed next phase after the module works).
