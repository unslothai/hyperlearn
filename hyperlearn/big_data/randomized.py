
from ..linalg import lu, svd, qr, eig
from numpy import random as _random

__all__ = ['randomizedSVD']


def randomized_projection(X, k, solver = 'lu', max_iter = 4):
	"""
	Projects X onto some random eigenvectors, then using a special
	variant of Orthogonal Iteration, finds the closest orthogonal
	representation for X.
	
	Solver can be QR or LU or None.
	"""
	if max_iter == 'auto':
		max_iter = 7 if k < 0.1*min(X.shape) else 4
		
	Q = _random.normal(size = (X.shape[1], k) ).astype(X.dtype)
	XT = X.T

	_solver =                       lambda x: lu(x)[0]
	if solver == 'qr': _solver =    lambda x: qr(x)[0]
	elif solver == None: _solver =  lambda x: x

	for __ in range(max_iter): 
		Q = _solver(XT @ _solver(X @ Q))

	return qr(X @ Q)[0]



def randomizedSVD(X, n_components = 2, max_iter = 'auto', solver = 'lu'):
	"""
	HyperLearn's Fast Randomized SVD is approx 10 - 30 % faster than
	Sklearn's implementation depending on n_components and max_iter.
	
	Uses NUMBA Jit accelerated functions when available, and tries to
	reduce memory overhead by chaining operations.
	
	Uses QR, LU or no solver to find the best SVD decomp. QR is most stable,
	but can be 2x slower than LU.    
	
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
	k = n_components
	n,p = X.shape
	transpose = (n >= p)
	X = X if transpose else X.T

	Q = randomized_projection(X, k, solver, max_iter)
	
	U, S, VT = svd(Q.T @ X, U_decision = ~transpose, transpose = True)
	return U, S, VT



def randomizedEig(X, n_components = 2, max_iter = 'auto', solver = 'lu'):
	"""
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
	k = n_components
	n,p = X.shape
	transpose = (n >= p)
	X = X if transpose else X.T

	Q = randomized_projection(X, k, solver, max_iter)
	
	S2, V = eig(Q.T @ X, U_decision = ~transpose)
	return S2, V

