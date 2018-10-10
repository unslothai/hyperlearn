
from numba import njit, prange, jit
from numpy import zeros, uint8, uint16, uint32, uint64, float32, float64
from numpy import sum as _sum
from warnings import warn as Warn


def getDtype(p, size):
	dtype = uint64
	if uint8(p) == p: dtype = uint8
	elif uint16(p) == p: dtype = uint16
	elif uint32(p) == p: dtype = uint32
	return zeros(size, dtype = dtype)


@njit(fastmath = True, nogil = True, cache = True)
def determine_nnz(X, rowCount):
	boolMat = (X != 0)
	nnz = 0
	n,p = X.shape
	
	for i in range(n):
		currNNZ = 0
		for j in range(p):
			currNNZ += boolMat[i, j]
		nnz += currNNZ
		rowCount[i] = currNNZ
	
	return boolMat, rowCount, nnz


def create_csr(X, boolMat, rowCount, nnz, temp):
	val = zeros(nnz, dtype = X.dtype)
	rowIndices = zeros(n+1, dtype = temp.dtype)
	colPointer = zeros(nnz, dtype = rowCount.dtype)
	
	p = X.shape[1]
	colIndex = zeros(p, dtype = rowCount.dtype)
	k = 0
	for i in range(p):
		colIndex[i] = k
		k += 1
	#colIndex = np.arange(X.shape[1]).astype(rowCount.dtype)
	
	k = 0
	for i in range(n):
		a = rowCount[i]
		rowIndices[i] += k
		k += a
	rowIndices[n] = nnz

	# val[left:right] = X[i][b]
	# colPointer[left:right] = colIndex[b]
	for i in prange(n):
		b = boolMat[i]
		left = rowIndices[i]
		right = rowIndices[i+1]
		
		Xib = X[i][b]
		colI = colIndex[b]
		k = 0
		for j in range(left, right):
			val[j] = Xib[k]
			colPointer[j] = colI[k]
			k += 1

	return val, colPointer, rowIndices
create_csr_cache = njit(create_csr, fastmath = True, nogil = True, cache = True)
create_csr_parallel = njit(create_csr, fastmath = True, nogil = True, parallel = True)


@njit(fastmath = True, nogil = True, cache = True)
def sum_A(val, colPointer, n, p):
	S = 0
	for i in range(len(val)):
		S += val[i]
	return S


@njit(fastmath = True, nogil = True, cache = True)
def sum_0(val, colPointer, n, p):
	S = zeros(p, dtype = val.dtype)
	
	for i in range(len(val)):
		S[colPointer[i]] += val[i]
	return S


@njit(fastmath = True, nogil = True, cache = True)
def sum_1(val, rowIndices, n, p):
	S = np.zeros(n, dtype = val.dtype)
	
	for i in range(n):
		left = rowIndices[i]
		right = rowIndices[i+1]
		S[i] += _sum(val[left:right])
	return S


@njit(fastmath = True, nogil = True, cache = True)
def mean_A(val, rowIndices, n, p):
	S = sum_A(val, rowIndies)
	return S/len(val)


@njit(fastmath = True, nogil = True, cache = True)
def mean_0(val, colPointer, n, p):
	A = zeros(p, dtype = val.dtype)
	
	nnz = len(val)
	for i in range(nnz):
		j = colPointer[i]
		A[j] += val[i]
		
	for i in range(p):
		if A[i] > 0:
			A[i] /= n
	return A


@njit(fastmath = True, nogil = True, cache = True)
def mean_1(val, rowIndices, n, p):
	A = zeros(n, dtype = val.dtype)
	
	for i in range(n):
		left = rowIndices[i]
		right = rowIndices[i+1]
		A[i] += _sum(val[left:right])
		if A[i] > 0:
			A[i] /= p
	return A


@njit(fastmath = True, nogil = True, cache = True)
def add_A(val, rowIndices, addon, n, p, copy = True):
	V = val.copy() if copy else val

	for i in range(len(val)):
		V[i] += addon
	return V, addon


@njit(fastmath = True, nogil = True, cache = True)
def add_0(val, colPointer, addon, n, p, copy = True):
	V = val.copy() if copy else val
	
	for i in range(len(val)):
		j = colPointer[i]
		V[i] += addon[j]
	return V, addon
	

@njit(fastmath = True, nogil = True, cache = True)
def add_1(val, rowIndices, addon, n, p, copy = True):
	V = val.copy() if copy else val

	for i in range(n):
		left = rowIndices[i]
		right = rowIndices[i+1]
		for j in range(left, right):
			V[j] += addon[i]
	return V, addon


@njit(fastmath = True, nogil = True, cache = True)
def div_A(val, colPointer, divisor, n, p, copy = True):
	V = val.copy() if copy else val

	for i in range(len(val)):
		V[i] /= divisor
	return V


@njit(fastmath = True, nogil = True, cache = True)
def div_0(val, colPointer, divisor, n, p, copy = True):
	V = val.copy() if copy else val
	
	for i in range(len(val)):
		j = colPointer[i]
		V[i] /= divisor[j]
	return V


@njit(fastmath = True, nogil = True, cache = True)
def div_1(val, rowIndices, divisor, n, p, copy = True):
	V = val.copy() if copy else val

	for i in range(n):
		left = rowIndices[i]
		right = rowIndices[i+1]
		for j in range(left, right):
			V[j] /= divisor[i]
	return V


@njit(fastmath = True, nogil = True, cache = True)
def mult_A(val, colPointer, mult, n, p, copy = True):
	V = val.copy() if copy else val

	for i in range(len(val)):
		V[i] *= mult
	return V


@njit(fastmath = True, nogil = True, cache = True)
def mult_0(val, colPointer, mult, n, p, copy = True):
	V = val.copy() if copy else val
	
	for i in range(len(val)):
		j = colPointer[i]
		V[i] *= mult[j]
	return V


@njit(fastmath = True, nogil = True, cache = True)
def mult_1(val, rowIndices, mult, n, p, copy = True):
	V = val.copy() if copy else val

	for i in range(n):
		left = rowIndices[i]
		right = rowIndices[i+1]
		for j in range(left, right):
			V[j] *= mult[i]
	return V


def CreateCSR(X, n_jobs = 1):
	n,p = X.shape
	rowCount = getDtype(p, n)

	boolMat, rowCount, nnz = determine_nnz(X, rowCount)

	if nnz/(n*p) > 0.8:
		Warn("Created sparse matrix has just under 20% zeros. Not a good idea to sparsify the matrix.")

	temp = getDtype(nnz, 1)

	f = create_csr_cache if n_jobs == 1 else create_csr_parallel
	return f(X, boolMat, rowCount, nnz, temp)

