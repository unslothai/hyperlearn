

from ..utils import _XXT, rowSum, reflect, setDiagonal
from numpy import zeros, newaxis
from numba import njit, prange
from ..sparse.csr import div_1 ,mult_1, _XXT as _XXT_sparse, rowSum as rowSum_sparse
from ..sparse.tcsr import _XXT as _XXT_triangular



def cosine_sim_triangular(N, D):
	"""
	[Added 21/10/2018]
	Quickly performs X / norm_rows / norm_rows.T on the TCSR matrix.
	"""
	n = len(N)
	move = 0
	
	# loop *-2 and adds S[:, newaxis]
	for i in prange(n-1):
		i1 = i+1
		
		left = i*i1 // 2
		s = N[i1]
		for j in range(left, left+i1):			
			# div N[:, newaxis]
			D[j] /= s
	
	# loop div N[newaxis, :]
	for a in prange(n-1):
		s = N[a]

		for b in range(a, n-1):
			# div N[newaxis, :] or N
			c = b*(b+1) // 2 + a
			D[c] /= s
	return D
cosine_sim_triangular_single = njit(cosine_sim_triangular, fastmath = True, nogil = True, cache = True)
cosine_sim_triangular_parallel = njit(cosine_sim_triangular, fastmath = True, nogil = True, parallel = True)



@njit(fastmath = True, nogil = True, cache = True)
def cosine_dis(XXT):
	"""
	[Added 22/10/2018]
	Performs XXT*-1 + 1 quickly on the lower triangular part.
	"""
	n = len(XXT)
	for i in range(n):
		for j in range(i):
			XXT[i, j] *= -1
			XXT[i, j] += 1
	return XXT



@njit(fastmath = True, nogil = True, cache = True)
def cosine_dis_triangular(D):
	"""
	[Added 22/10/2018]
	Performs XXT*-1 + 1 quickly on the TCSR.
	"""
	D *= -1
	D += 1
	return D



def cosine_similarity(X, Y = None, triangular = False, n_jobs = 1, copy = False):
	"""
	[Added 20/10/2018] [Edited 22/201/2018]
	[Edited 22/10/2018 Added Y option]
	Note: when using Y, speed improvement is approx 5% only from Sklearn.

	Cosine similarity is approx the same speed as Sklearn, but uses approx 10%
	less memory. One clear advantage is if you set triangular to TRUE, then it's faster.
	"""
	norm_rows = rowSum(X, norm = True)

	if Y is X:
		# Force algo to be triangular cosine rather than normal CS.
		Y = None

	if Y is None:
		if copy:
			XXT = _XXT(X.T)
			XXT /= norm_rows[:, newaxis]
			XXT /= norm_rows #[newaxis, :]
		else:
			XXT = _XXT(  (X/norm_rows[:, newaxis]).T  )

		if not triangular:
			XXT = reflect(XXT, n_jobs)

		# diagonal is set to 1
		setDiagonal(XXT, 1)
		return XXT
	else:
		D = X @ Y.T
		D /= norm_rows[:, newaxis]
		D /= rowSum(Y, norm = True)
		return D



def cosine_similarity_sparse(val, colPointer, rowIndices, n, p, triangular = False, dense_output = True,
	n_jobs = 1, copy = True):
	"""
	[Added 20/10/2018] [Edited 21/10/2018]
	Slightly faster than Sklearn's Cosine Similarity implementation.

	If dense_output is set to FALSE, then a TCSR Matrix (Triangular CSR Matrix) is
	provided and not a CSR matrix. This has the advantage of using only 1/2n^2 - n
	memory and not n^2 memory.
	"""
	norm_rows = rowSum_sparse(val, colPointer, rowIndices, norm = True)

	if dense_output:
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
		setDiagonal(XXT, 1)
	else:
		XXT = _XXT_triangular(val, colPointer, rowIndices, n, p, n_jobs)

		XXT = cosine_triangular_parallel(norm_rows, XXT) if n_jobs != 1 else \
			cosine_triangular_single(norm_rows, XXT)
	return XXT



def cosine_distances(X, Y = None, triangular = False, n_jobs = 1, copy = False):
	"""
	[Added 15/10/2018] [Edited 18/10/2018]
	[Edited 22/10/2018 Added Y option]
	Note: when using Y, speed improvement is approx 5-10% only from Sklearn.

	Slightly faster than Sklearn's Cosine Distances implementation.
	If you set triangular to TRUE, the result is much much faster.
	(Approx 50% faster than Sklearn)
	"""
	norm_rows = rowSum(X, norm = True)

	if Y is X:
		# Force algo to be triangular cosine rather than normal CS.
		Y = None

	if Y is None:
		if copy:
			XXT = _XXT(X.T)
			XXT /= norm_rows[:, newaxis]
			XXT /= norm_rows #[newaxis, :]
		else:
			XXT = _XXT(  (X/norm_rows[:, newaxis]).T  )

		# XXT*-1 + 1
		XXT = cosine_dis(XXT)

		if not triangular:
			XXT = reflect(XXT, n_jobs)

		# diagonal is set to 0 as zero distance between row i and i
		setDiagonal(XXT, 0)
		return XXT
	else:
		D = X @ Y.T
		D /= norm_rows[:, newaxis]
		D /= rowSum(Y, norm = True)
		D *= -1
		D += 1
		return D



def cosine_distances_sparse(val, colPointer, rowIndices, n, p, triangular = False, dense_output = True,
	n_jobs = 1, copy = True):
	"""
	[Added 22/10/2018]
	Slightly faster than Sklearn's Cosine Distances implementation.

	If dense_output is set to FALSE, then a TCSR Matrix (Triangular CSR Matrix) is
	provided and not a CSR matrix. This has the advantage of using only 1/2n^2 - n
	memory and not n^2 memory.
	"""
	norm_rows = rowSum_sparse(val, colPointer, rowIndices, norm = True)

	if dense_output:
		if copy:
			XXT = _XXT_sparse(val, colPointer, rowIndices, n, p, n_jobs)
			XXT /= norm_rows[:, newaxis]
			XXT /= norm_rows #[newaxis, :]
		else:
			val = div_1(val, colPointer, rowIndices, norm_rows, n, p, copy = False)
			XXT = _XXT_sparse(val, colPointer, rowIndices, n, p, n_jobs)
			val = mult_1(val, colPointer, rowIndices, norm_rows, n, p, copy = False)

		# XXT*-1 + 1
		XXT = cosine_dis(XXT)

		if not triangular: 
			XXT = reflect(XXT, n_jobs)

		# diagonal is set to 0 as zero distance between row i and i
		setDiagonal(XXT, 0)
	else:
		XXT = _XXT_triangular(val, colPointer, rowIndices, n, p, n_jobs)

		# XXT*-1 + 1
		XXT = cosine_dis_triangular(XXT)

		XXT = cosine_triangular_parallel(norm_rows, XXT) if n_jobs != 1 else \
			cosine_triangular_single(norm_rows, XXT)
	return XXT