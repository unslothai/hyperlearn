
from ..utils import _XXT, rowSum, reflect
from numpy import zeros, newaxis
from numba import njit, prange
from ..sparse.csr import _XXT as _XXT_sparse, rowSum as rowSum_sparse
from ..sparse.tcsr import _XXT as _XXT_triangular
from ..numba import maximum


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
	[Added 15/10/2018] [Edited 21/10/2018]
	Computes maxmimum(XXT, 0) faster. Much faster than Sklearn since uses 
	the notion that	distance(X, X) is symmetric.

	Steps:
		maximum(XXT, 0)
			Optimised. Instead of n^2 operations, does n(n-1)/2 operations.
	"""
	n = len(XXT)
	
	for i in prange(n):
		XXT[i, i] = 0
		for j in range(i):
			if XXT[i, j] < 0:
				XXT[i, j] = 0
			if not squared:
				XXT[i, j] **= 0.5
				
	return XXT
maximum0 = njit(_maximum0, fastmath = True, nogil = True, cache = True)
maximum0_parallel = njit(_maximum0, fastmath = True, nogil = True, parallel = True)



def euclidean_triangular(S, D, squared = False):
	"""
	[Added 21/10/2018]
	Quickly performs -2D + X^2 + X.T^2 on the TCSR matrix.
	Also applies maximum(D, 0) and then square roots distances
	if required.
	"""
	# Apply -2*D and add row-wise rowSum
	n = len(S)
	move = 0
	
	# loop *-2 and adds S[:, newaxis]
	for i in prange(n-1):
		i1 = i+1
		
		left = i*i1 // 2
		s = S[i1]
		for j in range(left, left+i1):
			# mult by -2
			D[j] *= -2
			
			# add S[:, newaxis]
			D[j] += s
	
	# loop adds S[newaxis, :]
	for a in prange(n-1):
		s = S[a]

		for b in range(a, n-1):
			# add S[newaxis, :] or S
			c = b*(b+1) // 2 + a
			D[c] += s
			
			# maximum(D, 0)
			if D[c] < 0:
				D[c] = 0
			if not squared:
				D[c] **= 0.5
	return D
euclidean_triangular_single = njit(euclidean_triangular, fastmath = True, nogil = True, cache = True)
euclidean_triangular_parallel = njit(euclidean_triangular, fastmath = True, nogil = True, parallel = True)



def euclidean_distances(X, Y = None, triangular = False, squared = False, n_jobs = 1):
	"""
	[Added 15/10/2018] [Edited 16/10/2018]
	[Edited 22/10/2018 Added Y option]
	Notice: parsing in Y will result in only 10% - 15% speed improvement, not 30%.

	Much much faster than Sklearn's implementation. Approx not 30% faster. Probably
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
	S = rowSum(X)
	if Y is X:
		# if X == Y, then defaults to fast triangular L2 distance algo
		Y = None

	if Y is None:
		XXT = _XXT(X.T)
		XXT = mult_minus2(XXT)
		
		XXT += S[:, newaxis]
		XXT += S #[newaxis,:]
		
		D = maximum0_parallel(XXT, squared) if n_jobs != 1 else maximum0(XXT, squared)
		if not triangular: 
			D = reflect(XXT, n_jobs)
	else:
		D = X @ Y.T
		D *= -2
		D += S[:, newaxis]
		D += rowSum(Y)
		D = maximum(D, 0)
		if not squared:
			D **= 0.5
	return D


def euclidean_distances_sparse(val, colPointer, rowIndices, n, p, triangular = False, dense_output = True,
	squared = False, n_jobs = 1):
	"""
	[Added 15/10/2018] [Edited 21/10/2018]
	Much much faster than Sklearn's implementation. Approx not 60% faster. Probably
	even faster if using n_jobs = -1 (actually 73% faster). [n = 10,000 p = 1,000]
	Uses the idea that distance(X, X) is symmetric,	and thus algorithm runs only on 
	1/2 triangular part. Also notice memory usage is now 60% better than Sklearn.

	If dense_output is set to FALSE, then a TCSR Matrix (Triangular CSR Matrix) is
	provided and not a CSR matrix. This has the advantage of using only 1/2n^2 - n
	memory and not n^2 memory.

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
	if dense_output:
		XXT = _XXT_sparse(val, colPointer, rowIndices, n, p, n_jobs)

		XXT = mult_minus2(XXT)
		S = rowSum_sparse(val, colPointer, rowIndices)

		XXT += S[:, newaxis]
		XXT += S #[newaxis,:]
		
		XXT = maximum0_parallel(XXT, squared) if n_jobs != 1 else maximum0(XXT, squared)
		if not triangular: 
			XXT = reflect(XXT, n_jobs)
	else:
		XXT = _XXT_triangular(val, colPointer, rowIndices, n, p, n_jobs)
		S = rowSum_sparse(val, colPointer, rowIndices)

		XXT = euclidean_triangular_parallel(S, XXT, squared = squared) if n_jobs != 1 else \
			euclidean_triangular_single(S, XXT, squared = squared)

	return XXT

