
hyperlearn.base
======================================
The base module contains several decorators and methods.

Example code:
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



