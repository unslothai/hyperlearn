
hyperlearn.linalg
======================================
The linalg module contains all mathematical methods, decompositions and mirrors both Numpy's linalg and Scipy's linalg modules. HyperLearn's modules are all optimized and I also showcase some novel new algorithms.

Matrix Decompositions
--------------------------
+-----------------------------+----------------------------------------------+
| Cholesky Decomposition      | cholesky(X, [alpha])						 |
+-----------------------------+----------------------------------------------+
| LU Decomposition            | lu(X, [L_only, U_only, overwrite])			 |
+-----------------------------+----------------------------------------------+
| Singular Value Decomposition| svd(X, [U_decision, overwrite])				 |
+-----------------------------+----------------------------------------------+
| QR Decomposition 			  | qr(X, [Q_only, R_only, overwrite])			 |
+-----------------------------+----------------------------------------------+