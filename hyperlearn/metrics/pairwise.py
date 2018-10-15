
from ..utils import _XXT
from numpy import zeros, newaxis
from numba import njit, prange


def _L2_dist_1(X, XXT):
	"""
	[Added 15/10/2018]
	Computes rowsum(X^2) quickly.
	Notice much faster than Sklearn since uses the notion that
	distance(X, X) is symmetric.

	Steps:
		XXT *= -2
			Optimised. Instead of n^2 operations, does n(n-1)/2 operations.
		S = rowsum(X^2)
	"""
	n, p = X.shape
	S = zeros(n, dtype = X.dtype) 
	
	for i in prange(n):
		for j in range(i-1):
			XXT[i, j] *= -2

	for i in prange(n):
		s = 0
		Xi = X[i]
		for j in range(p):
			Xij = Xi[j]
			s += Xij*Xij
		S[i] = s
		
	return S, XXT
L2_dist_1 = njit(_L2_dist_1, fastmath = True, nogil = True, cache = True)
L2_dist_1_parallel = njit(_L2_dist_1, fastmath = True, nogil = True, parallel = True)


def _L2_dist_2(XXT, tril = False, squared = False):
	"""
	[Added 15/10/2018]
	Computes maxmimum(XXT, 0) faster and reflects lower triangular to upper.
	Notice much faster than Sklearn since uses the notion that
	distance(X, X) is symmetric.

	Steps:
		maximum(XXT, 0)
			Optimised. Instead of n^2 operations, does n(n-1)/2 operations.
		reflect tril to triu
			Optimised. Instead of n^2 operations, does n(n-1)/2 operations.
	"""
	n = len(XXT)

	if squared == False:
		for i in prange(n):
			for j in range(i-1):
				if XXT[i, j] < 0:
					XXT[i, j] = 0
					XXT[i, j]*=0.5
	else:
		for i in prange(n):
			for j in range(i-1):
				if XXT[i, j] < 0:
					XXT[i, j] = 0
				
	if tril == False:
		for i in prange(n):
			XXT[i, i] = 0
			for j in range(i+1, n):
				XXT[i, j] = XXT[j, i]
	return XXT
L2_dist_2 = njit(_L2_dist_2, fastmath = True, nogil = True, cache = True)
L2_dist_2_parallel = njit(_L2_dist_2, fastmath = True, nogil = True, parallel = True)


def L2_dist(X, tril = False, squared = False, n_jobs = 1):
	"""
	[Added 15/10/2018]
	Much much faster than Sklearn's implementation. Approx ~30% faster. Probably
	even faster if using n_jobs = -1. Uses the idea that distance(X, X) is symmetric,
	and thus algorithm runs only on 1/2 triangular part.

	Old complexity:
		X @ XT 			n^2p
		rowSum(X^2)		np	
		XXT*-2			n^2
		XXT+X^2			2n^2
		maximum(XXT,0)	n^2
						n^2p + 4n^2 + np
	New complexity:
		sym X @ XT 		n^2p/2
		rowSum(X^2)		np	
		sym XXT*-2		n^2/2	
		sym XXT+X^2		n^2
		maximum(XXT,0)	n^2/2
						n^2p/2 + 2n^2 + np

	So New complexity approx= 1/2(Old complexity)
	"""
	XXT = _XXT(X.T)
	if n_jobs == 1:
		S, XXT = L2_dist_1(X, XXT)
	else:
		S, XXT = L2_dist_1_parallel(X, XXT)
	
	XXT += S[:, newaxis]
	XXT += S[newaxis,:]
	
	if n_jobs == 1:
		XXT = L2_dist_2(XXT, tril, squared)
	else:
		XXT = L2_dist_2_parallel(XXT, tril, squared)
	
	return XXT
	