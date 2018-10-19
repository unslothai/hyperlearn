

from ..utils import _XXT, _XXT_sparse, rowSum, rowSum_sparse, reflect
from numpy import zeros, newaxis
from numba import njit, prange
from ..sparse.csr import div_1 ,mult_1


def cosine_similarity(X, triangular = False, n_jobs = 1, copy = True):
	"""
	[Added 15/10/2018] [Edited 18/10/2018]
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
	norm_rows = rowSum(X, norm = True)

	if copy:
		XXT = _XXT(X.T)
		XXT /= norm_rows[:, newaxis]
		XXT /= norm_rows #[newaxis, :]
	else:
		XXT = _XXT(  (X/norm_rows[:, newaxis]).T  )

	if not triangular: 
		XXT = reflect(XXT, n_jobs)

	# diagonal is set to 1
	XXT.flat[::len(XXT)+1] = 1
	return XXT


def cosine_similarity_sparse(val, colPointer, rowIndices, n, p, triangular = False, n_jobs = 1, copy = True):
	"""
	[Added 15/10/2018] [Edited 18/10/2018]
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
	norm_rows = rowSum_sparse(val, colPointer, rowIndices, norm = True)

	if copy:
		XXT = _XXT_sparse(val, colPointer, rowIndices, n, p, n_jobs)
		XXT /= norm_rows[:, newaxis]
		XXT /= norm_rows #[newaxis, :]
	else:
		val = div_1(val, colPointer, rowIndices, norm_rows, n, p, copy = False)
		XXT = _XXT_sparse(val, colPointer, rowIndices, n, p, n_jobs)
		val = mult_1(val, colPointer, rowIndices, norm_rows, n, p, copy = False)

	if not triangular: 
		XXT = reflect(XXT, n_jobs)

	# diagonal is set to 1
	XXT.flat[::len(XXT)+1] = 1
	return XXT

