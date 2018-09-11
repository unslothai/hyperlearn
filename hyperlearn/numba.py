
from numpy import ones, eye, float32, float64, \
				sum as __sum, arange as _arange, sign as _sign, uint as _uint
from numpy.linalg import svd as _svd, pinv as _pinv, eigh as _eigh, \
					cholesky as _cholesky, lstsq as _lstsq, qr as _qr, \
					norm as _norm
from numba import njit

__all__ = ['svd', 'pinv', 'eigh', 'cholesky', 'lstsq', 'qr','norm',
			'mean', '_sum', 'sign', 'arange']


@njit(fastmath = True, nogil = True, cache = True)
def svd(X):
	return _svd(X, full_matrices = False)


@njit(fastmath = True, nogil = True, cache = True)
def pinv(X):
	return _pinv(X)


@njit(fastmath = True, nogil = True, cache = True)
def eigh(XTX):
	return _eigh(XTX)


@njit(fastmath = True, nogil = True, cache = True)
def cholesky(XTX):
	return _cholesky(XTX)


@njit(fastmath = True, nogil = True, cache = True)
def lstsq(X, y):
	return _lstsq(X, y.astype(X.dtype))[0]


@njit(fastmath = True, nogil = True, cache = True)
def qr(X):
	return _qr(X)


@njit(fastmath = True, nogil = True, cache = True)
def norm(v, d = 2):
	return _norm(v, d)


@njit(fastmath = True, nogil = True, cache = True)
def _0mean(X, axis = 0):
	return __sum(X, axis)/X.shape[axis]


def mean(X, axis = 0):
	if axis == 0 and X.flags['C_CONTIGUOUS']:
		return _0mean(X)
	else:
		return X.mean(axis)


@njit(fastmath = True, nogil = True, cache = True)
def sign(X):
	return _sign(X)


@njit(fastmath = True, nogil = True, cache = True)
def arange(i):
	return _arange(i)


@njit(fastmath = True, nogil = True, cache = True)
def _sum(X, axis = 0):
	return __sum(X, axis)


## TEST
print("""Note that first time import of HyperLearn will be slow, """
		"""since NUMBA code has to be compiled to machine code """
		"""for optimization purposes.""")

y32 = ones(2, dtype = float32)
y64 = ones(2, dtype = float64)

X = eye(2, dtype = float32)
A = svd(X)
A = eigh(X)
A = cholesky(X)
A = pinv(X)
A = lstsq(X, y32)
A = lstsq(X, y64)
A = qr(X)
A = norm(y32)
A = norm(y64)
A = mean(X)
A = mean(y32)
A = mean(y64)
A = _sum(X)
A = _sum(y32)
A = _sum(y64)
A = sign(X)
A = arange(100)

X = eye(2, dtype = float64)
A = svd(X)
A = eigh(X)
A = cholesky(X)
A = pinv(X)
A = lstsq(X, y32)
A = lstsq(X, y64)
A = qr(X)
A = norm(y32, 2)
A = norm(y64, 2)
A = mean(X, 1)
A = _sum(X)
A = sign(X)

A = None
X = None
y32 = None
y64 = None
