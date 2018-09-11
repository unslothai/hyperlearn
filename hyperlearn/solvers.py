
from .linalg import *
from .base import *
from .utils import _svdCond
from .numba import lstsq as _lstsq
from .big_data import LSMR


__all__ = ['solve', 'solveCholesky', 'solveSVD', 'solveEig', 'lstsq']


def solve(X, y, tol = 1e-6, condition_limit = 1e8, alpha = None):
	"""
	[NOTE: as of 12/9/2018, LSMR is default in HyperLearn, replacing the 2nd fastest Cholesky
	Solve. LSMR is 2-4 times faster, and uses N less memory]

	Implements extremely fast least squares LSMR using orthogonalization as seen in Scipy's LSMR and
	https://arxiv.org/abs/1006.0758 [LSMR: An iterative algorithm for sparse least-squares problems]
	by David Fong, Michael Saunders.

	Scipy's version of LSMR is surprisingly slow, as some slow design factors were used
	(ie np.sqrt(1 number) is slower than number**0.5, or min(a,b) is slower than using 1 if statement.)

	ALPHA is provided for regularization purposes like Ridge Regression.

	This algorithm works well for Sparse Matrices as well, and the time complexity analysis is approx:
		X.T @ y   * min(n,p) times + 3 or so O(n) operations
		==> O(np)*min(n,p)
		==> either min(O(n^2p + n), O(np^2 + n))

	This complexity is much better than Cholesky Solve which is the next fastest in HyperLearn.
	Cholesky requires O(np^2) for XT * X, then Cholesky needs an extra 1/3*O(np^2), then inversion
	takes another 1/3*(np^2), and finally (XT*y) needs O(np).

	So Cholesky needs O(5/3np^2 + np) >> min(O(n^2p + n), O(np^2 + n))

	So by factor analysis, expect LSMR to be approx 2 times faster or so.
	Interestingly, the Space Complexity is even more staggering. LSMR takes only maximum O(np^2) space
	for the computation of XT * y + some overhead.

	Cholesky requires XT * X space, which is already max O(n^2p) [which is huge].
	Essentially, Cholesky shines when P is large, but N is small. LSMR is good for large N, medium P
	"""

	if len(y.shape) > 1:
		if y.shape[1] > 1:
			print("LSMR can only work on single Ys. Try fitting 2 or more models.")
			return
	
	alpha = 0 if alpha is None else alpha

	good = True
	while ~good:
		theta_hat, good = LSMR(X, y, tol = tol, condition_limit = condition_limit, alpha = alpha)
		alpha = ALPHA_DEFAULT is alpha == 0 else alpha*10

	return theta_hat

	


def solveCholesky(X, y, alpha = None, fast = False):
	"""
	Computes the Least Squares solution to X @ theta = y using Cholesky
	Decomposition. This is the default solver in HyperLearn.
	
	Cholesky Solving is used as the 2nd default solver [as of 12/9/2018, default
	has been switched to LSMR (called solve)] in HyperLearn,
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



def solveSVD(X, y, alpha = None, fast = True):
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


