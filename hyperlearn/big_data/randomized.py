
from ..linalg import lu, svd, qr, eig
from numpy import random as _random, sqrt
from numpy.linalg import norm
from ..utils import _float, _svdCond, traceXTX, eig_flip, svd_flip
from ..random import uniform

# __all__ = ['randomizedSVD', 'randomizedEig']


def randomized_projection(X, k, solver = 'lu', max_iter = 4):
	"""
	[Edited 8/11/2018 Added QR Q_only parameter]
	Projects X onto some random eigenvectors, then using a special
	variant of Orthogonal Iteration, finds the closest orthogonal
	representation for X.
	
	Solver can be QR or LU or None.
	"""
	n, p = X.shape
	if max_iter == 'auto':
		# From Modern Big Data Algorithms --> seems like <= 4 is enough.
		_min = n if n <= p else p
		max_iter = 5 if k < 0.1 * _min else 4

	Q = uniform(-5, 5, p, int(k), X.dtype)
	XT = X.T

	_solver =                       lambda x: lu(x, L_only = True)
	if solver == 'qr': _solver =    lambda x: qr(x, Q_only = True)
	elif solver == None: _solver = 	lambda x: x

	for __ in range(max_iter):
		Q = _solver(XT @ _solver(X @ Q))

	return qr(X @ Q, Q_only = True)



def randomizedSVD(X, n_components = 2, max_iter = 'auto', solver = 'lu', n_oversamples = 10):
	"""
	[Edited 9/11/2018 Fixed SVD_flip]
	HyperLearn's Fast Randomized SVD is approx 10 - 30 % faster than
	Sklearn's implementation depending on n_components and max_iter.
	
	Uses NUMBA Jit accelerated functions when available, and tries to
	reduce memory overhead by chaining operations.
	
	Uses QR, LU or no solver to find the best SVD decomp. QR is most stable,
	but can be 2x slower than LU.

	****n_oversamples = 10. This follows Sklearn convention to increase the chance
							of more accurate SVD.
	
	References
	--------------
	* Sklearn's RandomizedSVD
	
	* Finding structure with randomness: Stochastic algorithms for constructing
	  approximate matrix decompositions
	  Halko, et al., 2009 http://arxiv.org/abs/arXiv:0909.4061

	* A randomized algorithm for the decomposition of matrices
	  Per-Gunnar Martinsson, Vladimir Rokhlin and Mark Tygert

	* An implementation of a randomized algorithm for principal component
	  analysis
	  A. Szlam et al. 2014
	"""
	n,p = X.shape
	transpose = (n < p)
	X = X if not transpose else X.T
	X = _float(X)

	Q = randomized_projection(X, n_components + n_oversamples, solver, max_iter)

	U, S, VT = svd(Q.T @ X, U_decision = transpose, transpose = True)
	U = Q @ U
	if transpose:
		U, VT = VT.T, U.T

	return U[:, :n_components], S[:n_components], VT[:n_components, :]



def randomizedEig(X, n_components = 2, max_iter = 'auto', solver = 'lu', n_oversamples = 10):
	"""
	[Edited 9/11/2018 Fixed Eig_Flip]
	HyperLearn's Randomized Eigendecomposition is an extension of Sklearn's
	randomized SVD. HyperLearn notices that the computation of U is not necessary,
	hence will use QR followed by SVD or just SVD depending on the situation.

	Likewise, solver = LU is default, and follows randomizedSVD
	
	References
	--------------
	* Sklearn's RandomizedSVD
	
	* Finding structure with randomness: Stochastic algorithms for constructing
	  approximate matrix decompositions
	  Halko, et al., 2009 http://arxiv.org/abs/arXiv:0909.4061

	* A randomized algorithm for the decomposition of matrices
	  Per-Gunnar Martinsson, Vladimir Rokhlin and Mark Tygert

	* An implementation of a randomized algorithm for principal component
	  analysis
	  A. Szlam et al. 2014
	"""
	n,p = X.shape
	transpose = (n < p)
	X = X if not transpose else X.T
	X = _float(X)

	Q = randomized_projection(X, n_components + n_oversamples, solver, max_iter)

	if transpose:
		V, S2, __ = svd(Q.T @ X, U_decision = transpose, transpose = True)
		V = Q @ V
		S2 **= 2
	else:
		S2, V = eig(Q.T @ X, U_decision = transpose)
	return S2[:n_components], V[:, :n_components]



def randomizedPinv(X, n_components = None, alpha = None):
	"""
	[Added 6/11/2018]
	Implements fast randomized pseudoinverse with regularization.
	Can be used as an approximation to the matrix inverse.
	"""
	if alpha != None: assert alpha >= 0
	alpha = 0 if alpha is None else alpha

	if n_components == None:
		# will provide approx sqrt(p) - 1 components.
		# A heuristic, so not guaranteed to work.
		k = int(sqrt(X.shape[1]))-1
		if k <= 0: k = 1
	else:
		k = int(n_components) if n_components > 0 else 1

	X = _float(X)

	U, S, VT = randomizedSVD(X, n_components)
	U, S, VT = _svdCond(U, S, VT, alpha)
	
	return VT.T * S @ U.T
	