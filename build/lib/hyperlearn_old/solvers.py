
from .linalg import *
from .base import *
from .utils import _svdCond, _float, _XTX, _XXT, fastDot
from .numba import lstsq as _lstsq
from .big_data import LSMR
from numpy import newaxis, cumsum, sqrt, hstack
from .big_data.randomized import randomizedSVD
from .big_data.truncated import truncatedEig


__all__ = ['solve', 'solveCholesky', 'solveSVD', 'solveEig', 'solvePartial', 'lstsq',
			'solveTLS']


def solve(X, y, tol = 1e-6, condition_limit = 1e8, alpha = None, weights = None, copy = False,
			non_negative = False, max_iter = None):
	"""
	[As of 12/9/2018, an optional non_negative argument is added. Note as accurate as Scipy's NNLS,
	by copies ideas from gradient descent.]

	[NOTE: as of 12/9/2018, LSMR is default in HyperLearn, replacing the 2nd fastest Cholesky
	Solve. LSMR is 2-4 times faster, and uses N less memory]

	>>> WEIGHTS is an array of Weights for Weighted / Generalized Least Squares [default None]
	>>> theta = (XT*W*X)^-1*(XT*W*y)

	Implements extremely fast least squares LSMR using orthogonalization as seen in Scipy's LSMR and
	https://arxiv.org/abs/1006.0758 [LSMR: An iterative algorithm for sparse least-squares problems]
	by David Fong, Michael Saunders.

	Scipy's version of LSMR is surprisingly slow, as some slow design factors were used
	(ie np.sqrt(1 number) is slower than number**0.5, or min(a,b) is slower than using 1 if statement.)

	ALPHA is provided for regularization purposes like Ridge Regression.
	
	Speed
	--------------
	This algorithm works well for Sparse Matrices as well, and the time complexity analysis is approx:
		X.T @ y   * min(n,p) times + 3 or so O(n) operations
		==> O(np)*min(n,p)
		==> either min(O(n^2p + n), O(np^2 + n))
		*** Note if Weights is present, complexity increases.
			Instead of fitting X^T*W*X, fits X^T/sqrt(W)
			So, O(np+n) is needed extra.

	This complexity is much better than Cholesky Solve which is the next fastest in HyperLearn.
	Cholesky requires O(np^2) for XT * X, then Cholesky needs an extra 1/3*O(np^2), then inversion
	takes another 1/3*(np^2), and finally (XT*y) needs O(np).

	So Cholesky needs O(5/3np^2 + np) >> min(O(n^2p + n), O(np^2 + n))

	So by factor analysis, expect LSMR to be approx 2 times faster or so.

	Memory
	--------------
	Interestingly, the Space Complexity is even more staggering. LSMR takes only maximum O(np^2) space
	for the computation of XT * y + some overhead.
		*** Note if Weights is present, and COPY IS TRUE, then memory is DOUBLED.
			Hence, try setting COPY to FALSE, memory will not change, and X will return back to its
			original state afterwards.

	Cholesky requires XT * X space, which is already max O(n^2p) [which is huge].
	Essentially, Cholesky shines when P is large, but N is small. LSMR is good for large N, medium P

	Weighted Least Squares
	------------------------
	Theta_hat = (XT * W * X)^-1 * (XT * y)
	In other words in gradient descent / iterative solves solve:
		X * sqrt(W) * theta_hat = y * sqrt(W)

		or: X*sqrt(W)  ==>  y*sqrt(W)
	"""

	if len(y.shape) > 1:
		if y.shape[1] > 1:
			print("LSMR can only work on single Ys. Try fitting 2 or more models.")
			return


	if weights is not None:
		if len(weights.shape) > 1:
			if weights.shape[1] > 1:
				print("Weights must be 1 dimensional.")
				return
		weights = weights.ravel()**0.5
		W = weights[:,newaxis]
		if copy:
			X = X*W
			y = y*weights
		else:
			X *= W
			y *= weights

	alpha = 0 if alpha is None else alpha

	good = False
	while not good:
		theta_hat, good = LSMR(X, y, tol = tol, condition_limit = condition_limit, alpha = alpha,
								non_negative = non_negative, max_iter = max_iter)
		alpha = ALPHA_DEFAULT if alpha == 0 else alpha*10

	# Return X back to its original state
	if not copy and weights is not None:
		X /= W
		y /= weights

	return theta_hat

	


def solveCholesky(X, y, alpha = None, fast = True):
	"""
	[Added 23/9/2018 added matrix multiplication decisions (faster multiply)
	 ie: if (XTX)^1(XTy) or ((XTX)^-1XT)y is faster]
	[Edited 20/10/2018 Major update - added LAPACK cholSolve --> 20% faster]
	[Edited 30/10/2018 Reduced RAM usage by clearing unused variables]

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
	unstable than double (float64). You might see stability problems if FAST = TRUE.
	Set it to FALSE if theres issues.
	"""
	n,p = X.shape
	X, y = _float(X), _float(y)
	XT = X.T
	covariance = _XTX(XT) if n >= p else _XXT(XT)
	
	cho = cholesky(covariance, alpha = alpha, fast = fast)
	del covariance; covariance = None; # saving memory


	if n >= p:
		# Use spotrs solve from LAPACK
		return cholSolve(cho, XT @ y, alpha = alpha)
	else:
		inv = invCholesky(cho, fast = fast)
		return fastDot(XT, inv, y)

	#return fastDot(inv, XT, y) if n >= p else fastDot(XT, inv, y)
	



def solveSVD(X, y, n_components = None, alpha = None, fast = True):
	"""
	[Edited 6/11/2018 Added n_components for Partial Solving]
	Computes the Least Squares solution to X @ theta = y using SVD.
	Slow, but most accurate. Specify n_components to reduce overfitting.
	Heurestic is 95% of variance is captured, if set to 'auto'.
	
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
	if alpha != None: assert alpha >= 0
	alpha = 0 if alpha is None else alpha
	X, y = _float(X), _float(y)

	U, S, VT = svd(X, fast = fast)
	U, S, VT = _svdCond(U, S, VT, alpha)

	if type(n_components) == float:
		if n_components > 1:
			n_components = int(n_components)

	if n_components == 'auto' or type(n_components) == int:
		# Notice heuristic of 90% variance explained.
		s = S / S.sum()
		s = cumsum(s)

		if n_components == 'auto':
			for i in range(len(s)):
				if s[i] >= 0.9: break
		else:
			i = n_components
		U, S, VT = U[:,:i], S[:i], VT[:i]

	return fastDot(VT.T * S,   U.T,   y)



def solvePartial(X, y, n_components = None, alpha = None, fast = True):
	"""
	[Added 6/11/2018]
	Computes the Least Squares solution to X @ theta = y using Randomized SVD.
	Much faster than normal SVD solving, and is not prone is overfitting.
	
	Optional alpha is used for regularization purposes.
	"""
	if alpha != None: assert alpha >= 0
	alpha = 0 if alpha is None else alpha

	if n_components == None or n_components == 'auto':
		# will provide approx sqrt(p) - 1 components.
		# A heuristic, so not guaranteed to work.
		k = int(sqrt(X.shape[1]))-1
		if k <= 0: k = 1
	else:
		k = int(n_components) if n_components > 0 else 1

	X, y = _float(X), _float(y)

	U, S, VT = randomizedSVD(X, k)
	U, S, VT = _svdCond(U, S, VT, alpha)
	
	return fastDot(VT.T * S,   U.T,   y)



def solveEig(X, y, alpha = None, fast = True):
	"""
	[Edited 30/10/2018 Reduced RAM usage by clearing unused variables]
	
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
	X, y = _float(X), _float(y)
	XT = X.T
	covariance = _XTX(XT) if n >= p else _XXT(XT)
	
	inv = pinvh(covariance, alpha = alpha, fast = fast)
	del covariance; covariance = None; # saving memory

	return fastDot(inv, XT, y) if n >= p else fastDot(XT, inv, y)



def lstsq(X, y):
	"""
	Returns normal Least Squares solution using LAPACK and Numba if
	installed. PyTorch will default to Cholesky Solve.
	"""
	X, y = _float(X), _float(y)
	return _lstsq(X, y)



def solveTLS(X, y, solver = 'truncated'):
	"""
	[Added 6/11/2018]
	Performs Total Least Squares based on the implementation in Wikipedia:
	https://en.wikipedia.org/wiki/Total_least_squares.
	The naming is rather deceptive, as it doesn't mean it'll yield better
	results than pure SVD solving.
	Normal linear regression assumes Y|X has gaussian noise. TLS assumes this
	AND X|Y has noise.

	Two solvers - full, truncated. Truncated is much much faster, as smallest
	eigen component is needed. Full solver uses Eigendecomposition, which is
	much much slower, but more accurate.
	"""
	p = X.shape[1]
	X, y = _float(X), _float(y)
	if len(y.shape) > 1:
		print("solveTLS works only on single column Ys.")
		return
	Z = hstack((X, y[:, newaxis]))

	if solver == 'full':
		__, V = eig(Z)

		VXY = V[:p, p]
		VYY = V[p:, p]
	else:
		# Uses truncatedEig
		V = truncatedEig(Z, 1, which = 'smallest')[1].flatten()
		VXY = V[:p]
		VYY = V[p]

	theta_hat = -VXY / VYY

	return theta_hat

