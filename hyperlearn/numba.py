
from numpy import ones, eye, float32, float64, \
				sum as __sum, arange as _arange, sign as __sign, uint as _uint, \
				abs as __abs, minimum as _minimum, maximum as _maximum
from numpy.linalg import svd as _svd, pinv as _pinv, eigh as _eigh, \
					cholesky as _cholesky, lstsq as _lstsq, qr as _qr, \
					norm as _norm
from numba import njit, prange
from .base import USE_NUMBA

__all__ = ['svd', 'pinv', 'eigh', 'cholesky', 'lstsq', 'qr','norm',
			'mean', '_sum', 'sign', 'arange', '_abs', 'minimum', 'maximum',
			'multsum', 'squaresum', '_sign']


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
	return __sign(X)


@njit(fastmath = True, nogil = True, cache = True)
def arange(i):
	return _arange(i)


@njit(fastmath = True, nogil = True, cache = True)
def _sum(X, axis = 0):
	return __sum(X, axis)


@njit(fastmath = True, nogil = True, cache = True)
def _abs(v):
	return __abs(v)


@njit(fastmath = True, nogil = True, cache = True)
def maximum(X, i):
    return _maximum(X, i)


@njit(fastmath = True, nogil = True, cache = True)
def minimum(X, i):
    return _minimum(X, i)


@njit(fastmath = True, nogil = True, cache = True)
def _min(a,b):
    if a < b:
        return a
    return b


@njit(fastmath = True, nogil = True, cache = True)
def _max(a,b):
    if a < b:
        return b
    return a


@njit(fastmath = True, nogil = True, cache = True)
def _sign(x):
    if x < 0:
        return -1
    return 1


@njit(fastmath = True, nogil = True, parallel = True)
def multsum(a,b):
    s = a[0]*b[0]
    for i in prange(1,len(a)):
        s += a[i]*b[i]
    return s


@njit(fastmath = True, nogil = True, parallel = True)
def squaresum(v):
	if len(v.shape) == 1:
	    s = v[0]**2
	    for i in prange(1,len(v)):
	        s += v[i]**2
	# else:

 #    return s


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
A = _abs(y32)
A = _abs(y64)
A = _abs(X)
A = _abs(10.0)
A = _abs(10)
A = minimum(X, 0)
A = minimum(y32, 0)
A = minimum(y64, 0)
A = maximum(X, 0)
A = maximum(y32, 0)
A = maximum(y64, 0)
A = _min(0,1)
A = _min(0.1,1.1)
A = _max(0,1)
A = _max(0.1,1.1)
A = multsum(y32, y32)
A = multsum(y32, y64)
A = multsum(y64, y64)
A = squaresum(y32)
A = squaresum(X)
A = squaresum(y64)
A = _sign(-1)
A = _sign(-1.2)
A = _sign(1.2)


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
A = _abs(X)
A = maximum(X, 0)
A = minimum(X, 0)
A = squaresum(X)


A = None
X = None
y32 = None
y64 = None
