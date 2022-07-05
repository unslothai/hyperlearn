
from numba import njit, prange
from numpy import zeros

"""
TCSR Matrix functions.
	1. _XXT
"""

def _XXT_triangular(val, colPointer, rowIndices, n, p):
	"""
	[Added 21/10/2018]
	Computes XXT and stores it as a triangular sparse matrix.
	A triangular sparse matrix removes the colPointer and rowIndices
	since the TCSR format assumes each element is in sucession. 

	This cuts memory total memory usage of the full dense matrix by 1/2
	[actually 1/2n^2 - n is used].
	"""
	size = n*(n-1) // 2 # floor division
	
	D = zeros(size, dtype = val.dtype)
	P = zeros(p, dtype = val.dtype)

	for k in prange(n-1):
		l = rowIndices[k]
		r = rowIndices[k+1]

		A = P.copy()
		b = l
		for i in range(r-l):
			x = colPointer[b]
			A[x] = val[b]
			b += 1
		
		for j in prange(k+1, n):
			l = rowIndices[j]
			r = rowIndices[j+1]
			s = 0
			c = l
			for a in range(r-l):
				z = colPointer[c]
				v = A[z]
				if v != 0:
					s += v*val[c]
				c += 1
				
			# Exact position in CSR is found via j(j-1)/2 + k
			D_where = j*(j-1) // 2 + k
			D[D_where] = s
	
	return D
_XXT_triangular_single = njit(_XXT_triangular, fastmath = True, nogil = True, cache = True)
_XXT_triangular_parallel = njit(_XXT_triangular, fastmath = True, nogil = True, parallel = True)


def _XXT(val, colPointer, rowIndices, n, p, n_jobs = 1):
	"""
	[Added 16/10/2018]
	Computes X @ XT very quickly, and stores it in a modified CSR matrix (Triangular CSR).
	Uses 1/2n^2 - n memory, and thus much more efficient and space conversing than if
	using a full CSR Matrix (memory reduced by approx 25 - 50%).
	"""
	XXT = _XXT_triangular_parallel(val, colPointer, rowIndices, n, p) if n_jobs != 1 else \
		_XXT_triangular_single(val, colPointer, rowIndices, n, p)
	return XXT


