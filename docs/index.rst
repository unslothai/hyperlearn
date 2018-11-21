
Welcome to HyperLearn!
======================================
HyperLearn aims to make Machine Learning algorithms run in at least 50% of their original time. Algorithms from Linear Regression to Principal Component Analysis are optimized by using LAPACK, BLAS, and parallelized through Numba.

Some key current achievements of HyperLearn:

1. 70% less time to fit Least Squares / Linear Regression than sklearn + 50% less memory usage
2. 50% less time to fit Non Negative Matrix Factorization than sklearn due to new parallelized algo
3. 40% faster full Euclidean / Cosine distance algorithms
4. 50% less time LSMR iterative least squares
5. New Reconstruction SVD - use SVD to impute missing data! Has .fit AND .transform. Approx 30% better than mean imputation
6. 50% faster Sparse Matrix operations - parallelized
7. RandomizedSVD is now 20 - 30% faster

Example code
**************
Singular Value Decomposition

.. code-block:: python

	import hyperlearn as hl
	U, S, VT = hl.linalg.svd(X)


Pseudoinverse of a matrix using Cholesky Decomposition

.. code-block:: python

	from hyperlearn.linalg import pinvc
	inv = pinvc(X)

	# check if pinv(X) * X = identity
	check = inv.dot(X)


QR Decomposition wanting only Q matrix

.. code-block:: python

	from hyperlearn import linalg
	Q = linalg.qr(X, Q_only = True)

	# check if Q == Q from full QR
	q, r = linalg.qr(X)


.. toctree::
   :maxdepth: 4

   base
   linalg
   utils
   license

Directory
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

