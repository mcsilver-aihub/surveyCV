# How surveyCV does cross-validation, in plain language

## The core idea

Ordinary k-fold CV shuffles individual rows into folds at random. That is fine
when every row is independent. Survey data is not: students come in schools,
schools come in groups (strata), and some students count for more than others
(weights). If you shuffle by row, two kids from the same school can land one in
the training fold and one in the test fold, and the model quietly cheats by
recognizing that school. Your CV score then looks better than the model really
is.

The fix is to make the folds respect how the data was collected. That is all
surveyCV does. It uses the three survey columns to build smarter folds.

![How surveyCV builds folds and cross-validates](surveyCV_cross_validation_diagram.png)

## The three survey parameters and what each one does

**PSU (the cluster, for example the school): keep it whole.**
A school is never split across folds. Every student from a given school goes into
the same fold together. So when a school is in the test fold, the model has never
seen any student from it during training. No cheating.

**Stratum (the group the schools were sampled within): keep folds balanced.**
Within each stratum, the schools are spread across all the folds, so every fold
ends up with a mix from every stratum. No fold is accidentally all rural schools
or all one region. Each fold looks like a small version of the whole survey.

**Weight (how many real students each surveyed student represents): used for
scoring.**
When the model is graded on the held-out fold, each student's error is weighted
by their survey weight. So the score reflects the population the survey was meant
to represent, not just the raw sample.

## How a single fold actually gets built (step by step)

1. Go stratum by stratum.
2. Inside a stratum, list its unique schools (PSUs) and shuffle them.
3. Deal those schools out across the folds like cards: one to fold 1, one to fold
   2, and so on around the table.
4. Every student inherits whichever fold their school landed in.

Repeat for all strata. Because you deal within each stratum, every fold gets
schools from every stratum (that is the balance), and because you deal whole
schools, no school is ever cut in half (that is the no-leak guarantee).

Then CV runs the usual way: hold out fold 1 as the test set, train on the rest,
score it; hold out fold 2, train on the rest, score it; and so on. The only
differences from normal CV are that the held-out chunk is whole schools instead
of random rows, and the score is weighted.

## A tiny concrete example

Say you want 5 folds, and a stratum has 10 schools. surveyCV shuffles those 10
schools and assigns 2 to each of the 5 folds. Every student in those 2 schools
goes wherever their school went. Do that for all 17 YRBS strata, and each fold
ends up holding roughly 2 schools from every stratum, with no school appearing in
more than one fold.

One catch worth knowing: a stratum can only fill as many folds as it has schools.
If some stratum has just 4 schools and you ask for 5 folds, that stratum cannot
reach all 5, so surveyCV warns you and you would drop to 4 folds. That is the same
limit the original paper hit on the NSFG data.

## Which function does which part

- `design_aware_folds(strata, clusters, n_folds)`: builds the fold labels using
  the rule above (the dealing-by-school-within-stratum part).
- `SurveyFold(...)`: the same thing wrapped so scikit-learn, GridSearchCV, or
  Optuna can use it as a drop-in `cv`.
- `cross_val_score_survey(..., weights=...)`: runs the train and score loop and
  applies the survey weights when grading each fold.

Summary: it builds folds by dealing out whole schools within each stratum
so no school is split and every fold mirrors the survey, then grades each fold
using the survey weights.
