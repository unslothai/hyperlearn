
from ..utils import _XXT
from numpy import zeros, newaxis
from numba import njit, prange


@njit(fastmath = True, nogil = True, cache = True)
def L2_dist_1(X, XXT):
	"""
	[Added 15/10/2018] [Edited 16/10/2018]
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
    
    for i in range(n):
        for j in range(i):
            XXT[i, j] *= -2

    for i in range(n):
        s = 0
        Xi = X[i]
        for j in range(p):
            Xij = Xi[j]
            s += Xij*Xij
        S[i] = s
    return S, XXT


def _L2_dist_2(XXT, tril = False):
	"""
	[Added 15/10/2018] [Edited 16/10/2018]
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
    
    for i in range(n):
        XXT[i, i] = 0
        for j in range(i):
            if XXT[i, j] < 0:
                XXT[i, j] = 0
    
    if tril == False:
        for i in range(n):
            for j in range(i, n):
                XXT[i, j] = XXT[j, i]
    return XXT
L2_dist_2 = njit(_L2_dist_2, fastmath = True, nogil = True, cache = True)
L2_dist_2_parallel = njit(_L2_dist_2, fastmath = True, nogil = True, parallel = True)


def _L2_XXT_sparse(val, colPointer, rowIndices, n, p):
	"""
	[Added 16/10/2018]
	Computes X @ XT very quickly. Approx 50-60% faster than Sklearn's version,
	as it doesn't optimize a lot. Note, computes only lower triangular X @ XT,
	and disregards diagonal (set to 0)
	"""
    D = zeros((n,n), dtype = X.dtype)
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
        
        for j in range(k+1, n):
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
L2_sparse = njit(_L2_XXT_sparse, fastmath = True, nogil = True, cache = True)
L2_sparse_parallel = njit(_L2_XXT_sparse, fastmath = True, nogil = True, parallel = True)


@njit(fastmath = True, nogil = True, cache = True)
def L2_sparse_1(val, colPointer, rowIndices, XXT):
	"""
	[Added 15/10/2018] [Edited 16/10/2018]
	Computes rowsum(X^2) quickly for Sparse Matrices
	Notice much faster than Sklearn since uses the notion that
	distance(X, X) is symmetric.
	"""
    n = len(XXT)
    S = zeros(n, dtype = X.dtype) 
    
    for a in range(n):
        for b in range(a):
            XXT[a, b] *= -2

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
    return S, XXT


def euclidean_distances(X, tril = False, n_jobs = 1):
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
    S, XXT = L2_dist_1(X, XXT)
    
    XXT += S[:, np.newaxis]
    XXT += S[np.newaxis,:]
    
    XXT = L2_dist_2_parallel(XXT, tril) if n_jobs != 1 else L2_dist_2(XXT, tril)
    return XXT


def euclidean_distances_sparse(val, colPointer, rowIndices, n, p, tril = False, n_jobs = 1):
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
    XXT = L2_sparse_parallel(val, colPointer, rowIndices, n, p) if n_jobs != 1 else \
        L2_sparse(val, colPointer, rowIndices, n, p)
    
    S, XXT = L2_sparse_1(val, colPointer, rowIndices, XXT)
    
    XXT += S[:, np.newaxis]
    XXT += S[np.newaxis,:]
    
    XXT = L2_dist_2_parallel(XXT, tril) if n_jobs != 1 else L2_dist_2(XXT, tril)
    return XXT

