
from ..utils import _XXT, _XXT_sparse, rowSum, rowSum_sparse, reflect
from numpy import zeros, newaxis
from numba import njit, prange


@njit(fastmath = True, nogil = True, cache = True)
def mult_minus2(XXT):
	"""
	[Added 17/10/2018]
	Quickly multiplies XXT by -2. Uses notion that XXT is symmetric,
	hence only lower triangular is multiplied.
	"""
	n = len(XXT)
	for i in range(n):
		for j in range(i):
			XXT[i, j] *= -2
	return XXT


def _maximum0(XXT, squared = True):
	"""
	[Added 15/10/2018] [Edited 18/10/2018]
	Computes maxmimum(XXT, 0) faster. Much faster than Sklearn since uses 
	the notion that	distance(X, X) is symmetric.

	Steps:
		maximum(XXT, 0)
			Optimised. Instead of n^2 operations, does n(n-1)/2 operations.
	"""
	n = len(XXT)
	
	if squared:
		for i in prange(n):
			XXT[i, i] = 0
			for j in range(i):
				if XXT[i, j] < 0:
					XXT[i, j] = 0
	else:
		for i in prange(n):
			XXT[i, i] = 0
			for j in range(i):
				if XXT[i, j] < 0:
					XXT[i, j] = 0
				XXT[i, j] **= 0.5
	return XXT
maximum0 = njit(_maximum0, fastmath = True, nogil = True, cache = True)
maximum0_parallel = njit(_maximum0, fastmath = True, nogil = True, parallel = True)


def euclidean_distances(X, triangular = False, squared = False, n_jobs = 1):
	"""
	[Added 15/10/2018] [Edited 16/10/2018]
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
	XXT = mult_minus2(XXT)
	S = rowSum(X)
	
	XXT += S[:, newaxis]
	XXT += S #[newaxis,:]
	
	XXT = maximum0_parallel(XXT, squared) if n_jobs != 1 else maximum0(XXT, squared)
	if not triangular: 
		XXT = reflect(XXT, n_jobs)
	return XXT


def euclidean_distances_sparse(val, colPointer, rowIndices, n, p, triangular = False, squared = False, n_jobs = 1):
	"""
	[Added 15/10/2018]
	Much much faster than Sklearn's implementation. Approx ~60% faster. Probably
	even faster if using n_jobs = -1 (actually 73% faster). [n = 10,000 p = 1,000]
	Uses the idea that distance(X, X) is symmetric,	and thus algorithm runs only on 
	1/2 triangular part. Also notice memory usage is now 60% better than Sklearn.

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
	XXT = _XXT_sparse(val, colPointer, rowIndices, n, p, n_jobs)

	XXT = mult_minus2(XXT)
	S = rowSum_sparse(val, colPointer, rowIndices)

	XXT += S[:, newaxis]
	XXT += S #[newaxis,:]
	
	XXT = maximum0_parallel(XXT, squared) if n_jobs != 1 else maximum0(XXT, squared)
	if not triangular: 
		XXT = reflect(XXT, n_jobs)
	return XXT

