

from numpy import uint8, uint16, uint32, uint64, float32, float64
from numpy import zeros, int8, int16, int32, int64, ndim
from warnings import warn as Warn
from numba import njit, prange


def getDtype(p, size, uint = True):
	"""
	Computes the exact best possible data type for CSR Matrix
	creation.
	"""
	p = int(1.25*p) # Just in case
	if uint:
		dtype = uint64
		if uint8(p) == p: dtype = uint8
		elif uint16(p) == p: dtype = uint16
		elif uint32(p) == p: dtype = uint32
		return zeros(size, dtype = dtype)
	else:
		dtype = int64
		if int8(p) == p: dtype = int8
		elif int16(p) == p: dtype = int16
		elif int32(p) == p: dtype = int32
		return zeros(size, dtype = dtype)


@njit(fastmath = True, nogil = True, cache = True)
def determine_nnz(X, rowCount):
	"""
	Uses close to no memory at all when computing how many non
	zeros are in the matrix. Notice the difference with Scipy
	is HyperLearn does NOT use nonzero(). This reduces memory
	usage dramatically.
	"""
	nnz = 0
	n,p = X.shape
	
	for i in range(n):
		currNNZ = 0
		Xi = X[i]
		for j in range(p):
			if Xi[j] != 0:
				currNNZ += 1
		nnz += currNNZ
		rowCount[i] = currNNZ
	return rowCount, nnz


def create_csr(X, rowCount, nnz, temp):
	"""
	[Added 10/10/2018] [Edited 13/10/2018]
	Before used extra memory keeping a Boolean Matrix (np bytes) and a
	ColIndex pointer which used p memory. Now, removed (np + p) memory usage,
	meaning larger matrices can be handled.

	Algorithm is 3 fold:

	1. Create RowIndices
	2. For every row in data:
		3. Store until a non 0 is seen.

	Algorithm takes approx O(n + np) time, which is similar to Scipy's.
	The only difference is now, parallelisation is possible, which can
	cut the time to approx O(n + np/c) where c = no of threads
	"""
	n = X.shape[0]
	val = zeros(nnz, dtype = X.dtype)
	rowIndices = zeros(n+1, dtype = temp.dtype)
	colPointer = zeros(nnz, dtype = rowCount.dtype)
	
	p = X.shape[1]
	
	k = 0
	for i in range(n):
		a = rowCount[i]
		rowIndices[i] += k
		k += a
	rowIndices[n] = nnz

	for i in prange(n):
		Xi = X[i]
		left = rowIndices[i]
		right = rowIndices[i+1]
		
		k = 0
		for j in range(left, right):
			while Xi[k] == 0:
				k += 1
			val[j] = Xi[k]
			colPointer[j] = k
			k += 1
	
	return val, colPointer, rowIndices
create_csr_cache = njit(create_csr, fastmath = True, nogil = True, cache = True)
create_csr_parallel = njit(create_csr, fastmath = True, nogil = True, parallel = True)



def CreateCSR(X, n_jobs = 1):
	"""
	[Added 10/10/2018] [Edited 13/10/2018]
	Much much faster than Scipy. In fact, HyperLearn uses less memory,
	by noticing indices >= 0, hence unsigned ints are used.

	Likewise, parallelisation is seen possible with Numba with n_jobs.
	Notice, an error message will be provided if 20% of the data is only zeros.
	It needs to be more than 20% zeros for CSR Matrix to shine.
	"""
	n,p = X.shape
	rowCount = getDtype(p, n)

	rowCount, nnz = determine_nnz(X, rowCount)

	if nnz/(n*p) > 0.8:
		Warn("Created sparse matrix has just under 20% zeros. Not a good idea to sparsify the matrix.")

	temp = getDtype(nnz, 1)

	f = create_csr_cache if n_jobs == 1 else create_csr_parallel
	return f(X, rowCount, nnz, temp)

