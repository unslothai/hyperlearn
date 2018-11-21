
hyperlearn.linalg
======================================
The linalg module contains all mathematical methods, decompositions and mirrors both Numpy's linalg and Scipy's linalg modules. HyperLearn's modules are all optimized and I also showcase some novel new algorithms.

Matrix Decompositions
--------------------------

+-------------------------------+---------------------------------------+-----------------+-------------------+
| Cholesky Decomposition        | cholesky(X, [alpha])                  | X = U.T @ U     | Symmetric Square  |
+-------------------------------+---------------------------------------+-----------------+-------------------+
| LU Decomposition              | lu(X, [L_only, U_only, overwrite])    | X = L @ U       | Any Matrix        |
+-------------------------------+---------------------------------------+-----------------+-------------------+
| Singular Value Decomposition  | svd(X, [U_decision, overwrite])       | X = U * S @ V.T | Any Matrix        |
+-------------------------------+---------------------------------------+-----------------+-------------------+
| QR Decomposition              | qr(X, [Q_only, R_only, overwrite])    | X = Q @ R       | Any Matrix        |
+-------------------------------+---------------------------------------+-----------------+-------------------+

Eigenvalue Problems
-----------------------

+-------------------------------+---------------------------------------+-----------------+-------------------+
| Symmetric EigenDecomposition  | eigh(X, [alpha, svd, overwrite])      | X = V * L @ V^-1| Symmetric Square  |
+-------------------------------+---------------------------------------+-----------------+-------------------+

Matrix Inversion
--------------------

+-------------------------------+---------------------------------------+-----------------+-------------------+
| Cholesky Inverse              | cho_inv(X, [turbo])                   | inv(X) @ X = I  | Symmetric Square  |
+-------------------------------+---------------------------------------+-----------------+-------------------+
| Pseudoinverse (Cholesky)      | pinvc(X, [alpha, turbo])              | inv(X) @ X = I  | Any Matrix        |
+-------------------------------+---------------------------------------+-----------------+-------------------+
| Symmetric Pseudoinverse       | pinvh(X, [alpha, turbo, n_jobs])      | inv(X) @ X = I  | Symmetric Square  |
+-------------------------------+---------------------------------------+-----------------+-------------------+
| Pseudoinverse (SVD)           | pinv(X, [alpha, overwrite])           | inv(X) @ X = I  | Any Matrix        |
+-------------------------------+---------------------------------------+-----------------+-------------------+
| Pseudoinverse (LU Decomp)     | pinvl(X, [alpha, turbo, overwrite])   | inv(X) @ X = I  | Square Matrix     |
+-------------------------------+---------------------------------------+-----------------+-------------------+