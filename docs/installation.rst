Installation
============

Requirements
------------

- Python 3.9 or newer
- numpy >= 1.21
- scikit-learn >= 1.0

Install from PyPI
-----------------

.. code-block:: bash

   pip install surveycv

Optional extras
---------------

.. code-block:: bash

   # test dependencies (pytest, pandas)
   pip install "surveycv[test]"

   # documentation dependencies (sphinx, theme)
   pip install "surveycv[docs]"

Install from source
-------------------

.. code-block:: bash

   git clone git@github.com:mcsilver-aihub/surveyCV.git
   cd surveyCV
   pip install -e ".[test,docs]"

Verify the install
------------------

.. code-block:: python

   import surveycv
   print(surveycv.__version__)
   print(surveycv.__all__)
