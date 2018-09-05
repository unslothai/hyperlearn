
from numpy import ones, eye, float32 as _float32, float64 as _float64
from numpy.linalg import svd as _svd, pinv as _pinv, eigh as _eigh, \
					cholesky as _cholesky, lstsq as _lstsq
from numba import njit

__all__ = ['numba_svd', 'numba_pinv', 'numba_eigh', 
			'numba_cholesky', 'numba_lstsq']


@njit(fastmath = True, nogil = True, cache = True)
def numba_svd(X):
	return _svd(X, full_matrices = False)


@njit(fastmath = True, nogil = True, cache = True)
def numba_pinv(X):
	return _pinv(X)


@njit(fastmath = True, nogil = True, cache = True)
def numba_eigh(XTX):
	return _eigh(XTX)


@njit(fastmath = True, nogil = True, cache = True)
def numba_cholesky(XTX):
	return _cholesky(XTX)


@njit(fastmath = True, nogil = True, cache = True)
def numba_lstsq(X, y):
	return _lstsq(X, y.astype(X.dtype))[0]

## TEST
y32 = ones(2, dtype = _float32)
y64 = ones(2, dtype = _float64)

X = eye(2, dtype = _float32)
A = numba_svd(X)
A = numba_eigh(X)
A = numba_cholesky(X)
A = numba_pinv(X)
A = numba_lstsq(X, y32)
A = numba_lstsq(X, y64)

X = eye(2, dtype = _float64)
A = numba_svd(X)
A = numba_eigh(X)
A = numba_pinv(X)
A = numba_cholesky(X)
A = numba_lstsq(X, y32)
A = numba_lstsq(X, y64)

A = None
X = None
y32 = None
y64 = None
