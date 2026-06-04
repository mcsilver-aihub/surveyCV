References and Citation
=======================

Citation
--------

If you use ``surveycv`` in published work, please cite the original methodology
paper:

.. code-block:: bibtex

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

References
----------

- Wieczorek, J., Guerin, C., & McMahon, T. (2022). K-fold cross-validation for
  complex sample surveys. *Stat*, 11(1), e454.
  https://doi.org/10.1002/sta4.454
- surveyCV R package (Wieczorek, Guerin, McMahon & Ratliff). CRAN.
  https://cran.r-project.org/package=surveyCV
- Wolter, K. M. (2007). *Introduction to Variance Estimation* (2nd ed.).
  Springer. Covers random groups, the group jackknife, and the cluster bootstrap
  for clustered survey data.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python.
  *Journal of Machine Learning Research*, 12, 2825-2830. Source of the
  cross-validator API that :class:`~surveycv.SurveyFold` follows.

License
-------

``surveycv`` is released under the MIT license. The methodology is due to
Wieczorek et al. (2022); please cite it. This package is an independent
clean-room implementation and is not affiliated with or endorsed by the original
authors.
