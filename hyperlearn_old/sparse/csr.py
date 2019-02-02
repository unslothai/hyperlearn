
from numba import njit, prange
from numpy import zeros, sum as _sum, array, hstack, searchsorted, ndim
from .base import getDtype

"""
CSR Matrix functions.
	1. sum [sum_A, sum_0, sum_1]
	2. mean [mean_A, mean_0, mean_1]
	3. mult [mult_A, mult_0, mult_1]
	4. div [div_A, div_0, div_1]
	5. add [add_A, add_0, add_1]
	6. min [min_A, min_0, min_1]
	7. max [max_A, max_0, max_1]
	8. diagonal [diagonal, diagonal_add]
	9. element [get_element]
	10. matmul [mat_vec, matT_vec, mat_mat]
	11. _XXT
	12. rowSum
"""


@njit(fastmath = True, nogil = True, cache = True)
def sum_A(val, colPointer, rowIndices, n, p):
	S = 0
	for i in range(len(val)):
		S += val[i]
	return S


@njit(fastmath = True, nogil = True, cache = True)
def sum_0(val, colPointer, rowIndices, n, p):
	S = zeros(p, dtype = val.dtype)
	
	for i in range(len(val)):
		S[colPointer[i]] += val[i]
	return S


@njit(fastmath = True, nogil = True, cache = True)
def sum_1(val, colPointer, rowIndices, n, p):
	S = np.zeros(n, dtype = val.dtype)
	
	for i in range(n):
		left = rowIndices[i]
		right = rowIndices[i+1]
		S[i] += _sum(val[left:right])
	return S


@njit(fastmath = True, nogil = True, cache = True)
def mean_A(val, colPointer, rowIndices, n, p):
	S = sum_A(val, rowIndies)
	return S/len(val)


@njit(fastmath = True, nogil = True, cache = True)
def mean_0(val, colPointer, rowIndices, n, p):
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
def mean_1(val, colPointer, rowIndices, n, p):
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
def div_A(val, colPointer, rowIndices, divisor, n, p, copy = True):
	V = val.copy() if copy else val

	for i in range(len(val)):
		V[i] /= divisor
	return V


@njit(fastmath = True, nogil = True, cache = True)
def div_0(val, colPointer, rowIndices, divisor, n, p, copy = True):
	V = val.copy() if copy else val
	
	for i in range(len(val)):
		j = colPointer[i]
		V[i] /= divisor[j]
	return V


@njit(fastmath = True, nogil = True, cache = True)
def div_1(val, colPointer, rowIndices, divisor, n, p, copy = True):
	V = val.copy() if copy else val

	for i in range(n):
		left = rowIndices[i]
		right = rowIndices[i+1]
		for j in range(left, right):
			V[j] /= divisor[i]
	return V


@njit(fastmath = True, nogil = True, cache = True)
def mult_A(val, colPointer, rowIndices, mult, n, p, copy = True):
	V = val.copy() if copy else val

	for i in range(len(val)):
		V[i] *= mult
	return V


@njit(fastmath = True, nogil = True, cache = True)
def mult_0(val, colPointer, rowIndices, mult, n, p, copy = True):
	V = val.copy() if copy else val
	
	for i in range(len(val)):
		j = colPointer[i]
		V[i] *= mult[j]
	return V


@njit(fastmath = True, nogil = True, cache = True)
def mult_1(val, colPointer, rowIndices, mult, n, p, copy = True):
	V = val.copy() if copy else val

	for i in range(n):
		left = rowIndices[i]
		right = rowIndices[i+1]
		for j in range(left, right):
			V[j] *= mult[i]
	return V


@njit(fastmath = True, nogil = True, cache = True)
def min_A(val, colPointer, rowIndices, n, p):
	M = 0
	for i in range(len(val)):
		v = V[i]
		if v < M:
			M = v
	return M


@njit(fastmath = True, nogil = True, cache = True)
def min_0(val, colPointer, rowIndices, n, p):
	M = zeros(p, dtype = val.dtype)
	
	for i in range(len(val)):
		v = V[i]
		coli = colPointer[i]
		if v < M[coli]:
			M[coli] = v
	return M


@njit(fastmath = True, nogil = True, cache = True)
def min_1(val, colPointer, rowIndices, n, p):
	M = zeros(n, dtype = val.dtype)
	
	for i in range(n):
		left = rowIndices[i]
		right = rowIndices[i+1]

		Mi = M[i]
		for j in val[left:right]:
			if j < Mi:
				Mi = j
		M[i] = Mi
	return M


@njit(fastmath = True, nogil = True, cache = True)
def max_A(val, colPointer, rowIndices, n, p):
	M = 0
	for i in range(len(val)):
		v = V[i]
		if v > M:
			M = v
	return M


@njit(fastmath = True, nogil = True, cache = True)
def max_0(val, colPointer, rowIndices, n, p):
	M = zeros(p, dtype = val.dtype)
	
	for i in range(len(val)):
		v = V[i]
		coli = colPointer[i]
		if v > M[coli]:
			M[coli] = v
	return M


@njit(fastmath = True, nogil = True, cache = True)
def max_1(val, colPointer, rowIndices, n, p):
	M = zeros(n, dtype = val.dtype)
	
	for i in range(n):
		left = rowIndices[i]
		right = rowIndices[i+1]

		Mi = M[i]
		for j in val[left:right]:
			if j > Mi:
				Mi = j
		M[i] = Mi
	return M


@njit(fastmath = True, nogil = True, cache = True)
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



@njit(fastmath = True, nogil = True, cache = True)
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
	added = zeros(n, dtype = minimum.dtype)
	move = 0
	
	# move = go move steps forward
	# goes from left --> right
	for i in range(size):
		a = A[i]
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
			newV[m] = a
			newV[m+1:r+1] = V[lk:right]      
			move += 1
		else:
			newC[l:r] = C[left:right]
			newV[l:r] = V[left:right]
			newV[l-k-1] += a # -1 and not + 1 since -k
		  
	# Update rest of matrix if n > size  
	i += 1
	if i < n:
		newV[R[i]+move:] = V[R[i]:]
		newC[R[i]+move:] = C[R[i]:]
		added[i:] = move
	
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
	R[n] += a # Add to end of rowPointer, since size(n+1)
	
	return newV, newC, R



def diagonal_add(val, colPointer, rowIndices, n, p, addon, copy = True):
	"""
	See _diagonal_add documentation.
	"""
	size = min(n, p)
	if ndim(addon) == 0:
		A = zeros(size, dtype = val.dtype)+addon
	else:
		A = addon
	assert len(A) == size
	minimum = minimum = getDtype(min(n,p), size = 1, uint = False)
	return _diagonal_add(val, colPointer, rowIndices, n, p, minimum, A, copy)



@njit(fastmath = True, nogil = True, cache = True)
def get_element(val, colPointer, rowIndices, n, p, i, j):
	"""
	[Added 14/10/2018]
	Get A[i,j] element. HyperLearn's algorithm is O(logp) complexity, which is much
	better than Scipy's O(p) complexity. HyperLearn uses binary search to find the
	element.
	"""
	assert i < n and j < p
	left = rowIndices[i]
	right = rowIndices[i+1]
	select = colPointer[left:right]
	search = searchsorted(select, j)
	
	if search < len(select):
		if select[search] == j:
			return val[left+search]
	return 0



def _mat_vec(val, colPointer, rowIndices, n, p, y):
	"""
	Added [13/10/2018]
	X @ y is found. Scipy & HyperLearn has similar speed. Notice, now
	HyperLearn can be parallelised! This reduces complexity to approx
	O(np/c) where c = no of threads / cores
	"""
	Z = zeros(n, dtype = y.dtype)
	
	for i in prange(n):
		s = 0
		for j in range(rowIndices[i], rowIndices[i+1]):
			s += val[j]*y[colPointer[j]]
		Z[i] = s
	return Z
mat_vec = njit(_mat_vec, fastmath = True, nogil = True, cache = True)
mat_vec_parallel = njit(_mat_vec, fastmath = True, nogil = True, parallel = True)


def _matT_vec(val, colPointer, rowIndices, n, p, y):
	"""
	Added [13/10/2018]
	X.T @ y is found. Notice how instead of converting CSR to CSC matrix, a direct
	X.T @ y can be found. Same complexity as mat_vec(X, y). Also, HyperLearn is
	parallelized, allowing for O(np/c) complexity.
	"""
	Z = zeros(p, dtype = y.dtype)

	for i in prange(n):
		yi = y[i]
		for j in range(rowIndices[i], rowIndices[i+1]):
			Z[colPointer[j]] += val[j]*yi
	return Z
matT_vec = njit(_matT_vec, fastmath = True, nogil = True, cache = True)
matT_vec_parallel = njit(_matT_vec, fastmath = True, nogil = True, parallel = True)



def _mat_mat(val, colPointer, rowIndices, n, p, X):
	"""
	Added [14/10/2018]
	A @ X is found where X is a dense matrix. Mostly the same as Scipy, albeit slightly faster.
	The difference is now, HyperLearn is parallelized, which can reduce times by 1/2 or more.
	"""
	K = X.shape[1]
	Z = zeros((n, K), dtype = X.dtype)

	for i in prange(n):
		left = rowIndices[i]
		right = rowIndices[i+1]
		
		for k in range(K):
			s = 0
			for j in range(left, right):
				s += val[j]*X[colPointer[j], k]
			Z[i, k] = s
	return Z
mat_mat = njit(_mat_mat, fastmath = True, nogil = True, cache = True)
mat_mat_parallel = njit(_mat_mat, fastmath = True, nogil = True, parallel = True)


def _matT_mat(val, colPointer, rowIndices, n, p, X):
	"""
	Added [14/10/2018]
	A.T @ X is found where X is a dense matrix. Mostly the same as Scipy, albeit slightly faster.
	The difference is now, HyperLearn is parallelized, which can reduce times by 1/2 or more.
	"""
	K = X.shape[1]
	A = zeros((K, p), dtype = X.dtype)
	zero = zeros(p, dtype = X.dtype)

	for k in prange(K):
		Z = zero.copy()
		y = X[:,k]
		
		for i in range(n):
			yi = y[i]
			for j in range(rowIndices[i], rowIndices[i+1]):
				Z[colPointer[j]] += val[j]*yi
		for i in range(p):
			A[k, i] = Z[i]
	return A.T.copy()
matT_mat = njit(_matT_mat, fastmath = True, nogil = True, cache = True)
matT_mat_parallel = njit(_matT_mat, fastmath = True, nogil = True, parallel = True)




def XXT_sparse(val, colPointer, rowIndices, n, p):
	"""
	See _XXT_sparse documentation.
	"""
	D = zeros((n,n), dtype = val.dtype)
	P = zeros(p, dtype = val.dtype)

	for k in prange(n-1):
		l = rowIndices[k]
		r = rowIndices[k+1]

		R = P.copy()
		b = l
		for i in range(r-l):
			x = colPointer[b]
			R[x] = val[b]
			b += 1
		
		for j in prange(k+1, n):
			l = rowIndices[j]
			r = rowIndices[j+1]
			s = 0
			c = l
			for a in range(r-l):
				z = colPointer[c]
				v = R[z]
				if v != 0:
					s += v*val[c]
				c += 1
			D[j, k] = s
	return D
_XXT_sparse_single = njit(XXT_sparse, fastmath = True, nogil = True, cache = True)
_XXT_sparse_parallel = njit(XXT_sparse, fastmath = True, nogil = True, parallel = True)


def _XXT(val, colPointer, rowIndices, n, p, n_jobs = 1):
	"""
	[Added 16/10/2018]
	Computes X @ XT very quickly. Approx 50-60% faster than Sklearn's version,
	as it doesn't optimize a lot. Note, computes only lower triangular X @ XT,
	and disregards diagonal (set to 0)
	"""
	XXT = _XXT_sparse_parallel(val, colPointer, rowIndices, n, p) if n_jobs != 1 else \
		_XXT_sparse_single(val, colPointer, rowIndices, n, p)
	return XXT



@njit(fastmath = True, nogil = True, cache = True)
def rowSum(val, colPointer, rowIndices, norm = False):
	"""
	[Added 17/10/2018]
	Computes rowSum**2 for sparse matrix efficiently, instead of using einsum
	"""
	n = len(rowIndices)-1
	S = zeros(n, dtype = val.dtype)

	for i in range(n):
		s = 0
		l = rowIndices[i]
		r = rowIndices[i+1]
		b = l
		for j in range(r-l):
			Xij = val[b]
			s += Xij*Xij
			b += 1
		S[i] = s
	if norm:
		S**=0.5
	return S

