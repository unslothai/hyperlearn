
from .linalg import *
from .base import *
from .numba import lstsq as _lstsq

__all__ = ['solveCholesky', 'solveSvd', 'solveEig', 'lstsq']


def solveCholesky(X, y, alpha = None, fast = False):
	"""
	Computes the Least Squares solution to X @ theta = y using Cholesky
	Decomposition. This is the default solver in HyperLearn.
	
	Cholesky Solving is used as the default solver in HyperLearn,
	as it is super fast and allows regularization. HyperLearn's
	implementation also handles rank deficient and ill-conditioned
	matrices perfectly with the help of the limiting behaivour of
	adding forced epsilon regularization.
	
	Optional alpha is used for regularization purposes.
	
	|  Method   |   Operations    | Factor * np^2 |
	|-----------|-----------------|---------------|
	| Cholesky  |   1/3 * np^2    |      1/3      |
	|    QR     |   p^3/3 + np^2  |   1 - p/3n    |
	|    SVD    |   p^3   + np^2  |    1 - p/n    |
	
	Speed
	--------------
	If USE_GPU:
		Uses PyTorch's Cholesky and Triangular Solve given identity matrix. 
		Speed is OK.
	If CPU:
		Uses Numpy's Fortran C based Cholesky.
		If NUMBA is not installed, uses very fast LAPACK functions.
		Also, uses very fast LAPACK algorithms for triangular system inverses.
	
	Stability
	--------------
	Note that LAPACK's single precision (float32) solver (strtri) is much more
	unstable than double (float64). So, default strtri is OFF.
	However, speeds are reduced by 50%.
	"""
	n,p = X.shape
	XT = X.T
	covariance = XT @ X if n >= p else X @ XT
	
	cho = cholesky(covariance, alpha = alpha, fast = fast)
	inv = invCholesky(cho, fast = fast)
	
	return inv @ (XT @ y) if n >= p else XT @ (inv @ y)



def solveSvd(X, y, alpha = None, fast = True):
	"""
	Computes the Least Squares solution to X @ theta = y using SVD.
	Slow, but most accurate.
	
	Optional alpha is used for regularization purposes.
	
	Speed
	--------------
	If USE_GPU:
		Uses PyTorch's SVD. PyTorch uses (for now) a NON divide-n-conquer algo.
		Submitted report to PyTorch:
		https://github.com/pytorch/pytorch/issues/11174
	If CPU:
		Uses Numpy's Fortran C based SVD.
		If NUMBA is not installed, uses divide-n-conqeur LAPACK functions.
	
	Stability
	--------------
	Condition number is:
		float32 = 1e3 * eps * max(S)
		float64 = 1e6 * eps * max(S)
	"""
	if alpha is not None: assert alpha >= 0
	alpha = 0 if alpha is None else alpha
	
	U, S, VT = svd(X, fast = fast)
	U, S, VT = _svdCond(U, S, VT, alpha)
	
	return (VT.T * S) @ (U.T @ y)



def solveEig(X, y, alpha = None, fast = True):
	"""
	Computes the Least Squares solution to X @ theta = y using
	Eigendecomposition on the covariance matrix XTX or XXT.
	Medium speed and accurate, where this lies between
	SVD and Cholesky.
	
	Optional alpha is used for regularization purposes.
	
	Speed
	--------------
	If USE_GPU:
		Uses PyTorch's EIGH. PyTorch uses (for now) a non divide-n-conquer algo.
		Submitted report to PyTorch:
		https://github.com/pytorch/pytorch/issues/11174
	If CPU:
		Uses Numpy's Fortran C based EIGH.
		If NUMBA is not installed, uses divide-n-conqeur LAPACK functions.
		Note Scipy's EIGH as of now is NON divide-n-conquer.
		Submitted report to Scipy:
		https://github.com/scipy/scipy/issues/9212
	
	Stability
	--------------
	Condition number is:
		float32 = 1e3 * eps * max(abs(S))
		float64 = 1e6 * eps * max(abs(S))
		
	Alpha is added for regularization purposes. This prevents system
	rounding errors and promises better convergence rates.
	"""
	n,p = X.shape
	XT = X.T
	covariance = XT @ X if n >= p else X @ XT
	
	inv = pinvh(covariance, alpha = alpha, fast = fast)

	return inv @ (XT @ y) if n >= p else XT @ (inv @ y)




def lstsq(X, y):
	"""
	Returns normal Least Squares solution using LAPACK and Numba if
	installed. PyTorch will default to Cholesky Solve.
	"""
	
	return _lstsq(X, y)

