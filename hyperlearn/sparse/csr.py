
from numba import njit, prange, jit
from numpy import zeros, uint8, uint16, uint32, uint64, float32, float64
from numpy import sum as _sum, array, hstack, searchsorted
from numpy import int8, int16, int32, int64, ndim
from warnings import warn as Warn


def getDtype(p, size, uint = True):
	"""
	Computes the exact best possible data type for CSR Matrix
	creation.
	"""
	p = 2*p # Just in case
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
		for j in range(p):
			currNNZ += (X[i, j] != 0)
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


@njit(fastmath = True, nogil = True)
def diagonal(val, colPointer, rowIndices, n, p):
	"""
	[Added 10/10/2018] [Edited 13/10/2018]
	Extracts the diagonal elements of a CSR Matrix. Note only gets
	square diagonal (not off-diagonal). HyperLearn's algorithm is faster
	than Scipy's, as it uses binary search, whilst Scipy uses Linear Search.
	
	d = min(n, p)
	HyperLearn = O(d log p)
	Scipy = O(dp)
	"""
	size = min(n,p)
	diag = zeros(size, dtype = val.dtype)
	for i in range(size):
		left = rowIndices[i]
		right = rowIndices[i+1]
		
		partial = colPointer[left:right]
		k = searchsorted(partial, i)
		
		# Get only diagonal elements else 0
		if partial[k] == i:
			diag[i] = val[left+k]
	return diag



@njit(fastmath = True, nogil = True)
def _diagonal_add(val, colPointer, rowIndices, n, p, minimum, addon, copy = True):
	"""
	[Added 10/10/2018] [Edited 13/10/2018]
	HyperLearn's add diagonal to CSR Matrix is optimized and uses Binary Search.
	This algorithm is executed in 4 distinct steps:
	
	1. Intialise K vector (kinda like Dijkstra's algorithm)
		used to log where the diagonal element is.
	2. Use binary search to find the diagonal element.
		Scipy's is linear search, which is slower.
	3. Change Val, ColPointer vectors.
	4. Change RowIndices vector.
	
	Steps 1->4 also are mostly optimized in time complexity.
	
		d = min(n, p)
	1. O(d)
	2. O(d log p)
	3. O(2 dp)
	4. O(n)
	
	The magic is step 2, where Scipy is O(dp). Likewise, total complexity
	is O(d + n + dlogp + dp)
	"""
	size = min(n,p)
	A = addon.astype(val.dtype)
	
	V = val
	C = colPointer
	R = rowIndices.copy() if copy else rowIndices
	
	extend = 0
	
	K = zeros(size, dtype = minimum.dtype)
	for i in range(size):
		K[i] -= 1

	for i in range(size):
		left = R[i]
		right = R[i+1]
		
		partial = C[left:right]
		# Get only diagonal elements else 0
		k = searchsorted(partial, i)
		
		if len(partial) != k:
			if partial[k] == i:
				K[i] = -k-1 # neg -1 as we don't want to mix
							# the true extra elements
			else:
				K[i] = k
				extend += 1
		else:
			K[i] = k
			extend += 1

	newN = len(C)+extend
	newC = zeros(newN, dtype = C.dtype)
	newV = zeros(newN, dtype = V.dtype)
	added = zeros(size, dtype = minimum.dtype)
	move = 0
	
	# move = go move steps forward
	# goes from left --> right
	for i in range(size):
		k = K[i]
		left = R[i]
		right = R[i+1]
		l = left+move
		r = right+move
		m = l+k
		lk = left+k
		
		added[i] = move
		if k > -1:
			newC[l:m] = C[left:lk]
			newC[m] = i
			newC[m+1:r+1] = C[lk:right]
			
			newV[l:m] = V[left:lk]
			newV[m] = A[i]
			newV[m+1:r+1] = V[lk:right]      
			move += 1
		else:
			newC[l:r] = C[left:right]
			newV[l:r] = V[left:right]
			newV[l-k-1] += A[i] # -1 and not + 1 since -k
	
	# Update rowPointer
	for i in range(n):
		a = added[i]
		if i < size:
			R[i] += a
		else:
			# Handles if n > d --> just pad the rowPointer
			for j in range(i, n):
				R[j] += a
			break
	R[i+1] += a # Add to end of rowPointer, since size(n+1)
	
	return newV, newC, R



def diagonal_add(val, colPointer, rowIndices, n, p, minimum, addon, copy = True):
	"""
	See _diagonal_add documentation.
	"""
	size = min(n, p)
	if ndim(addon) == 0:
		A = zeros(size, dtype = val.dtype)+addon
	else:
		A = addon
	assert len(A) == size
	return _diagonal_add(val, colPointer, rowIndices, n, p, minimum, A, copy)



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

