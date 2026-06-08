API Reference
=============

All public objects are importable directly from the top-level package:

.. code-block:: python

   from surveycv import (
       design_aware_folds,
       SurveyFold,
       survey_train_test_split,
       cross_val_score_survey,
       cluster_bootstrap_ci,
       ClusterBootstrapResult,
   )

.. currentmodule:: surveycv

Fold construction
-----------------

.. autofunction:: design_aware_folds

Cross-validator
---------------

.. autoclass:: SurveyFold
   :members:
   :show-inheritance:

Train/test split
----------------

.. autofunction:: survey_train_test_split

Weighted scoring
----------------

.. autofunction:: cross_val_score_survey

Confidence intervals
--------------------

.. autofunction:: cluster_bootstrap_ci

.. autoclass:: ClusterBootstrapResult
   :no-members:
