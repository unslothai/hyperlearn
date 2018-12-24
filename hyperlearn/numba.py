
from functools import wraps
from numba import njit, prange
import numpy as np
from numpy import linalg

###
def jit(f = None, parallel = False):
    """
    [Added 14/11/2018] [Edited 17/11/2018 Auto add n_jobs argument to functions]
    Decorator onto Numba NJIT compiled code.

    input:      1 argument, 1 optional
    ----------------------------------------------------------
    function:   Function to decorate
    parallel:   If true, sets cache automatically to false

    returns:    CPU Dispatcher function
    ----------------------------------------------------------
    """
    def decorate(f):
        if parallel:
            f_multi = njit(f, fastmath = True, nogil = True, parallel = True, cache = False)
        f_single = njit(f, fastmath = True, nogil = True, parallel = False, cache = True)
        
        @wraps(f)
        def wrapper(*args, **kwargs):
            # If n_jobs is an argument --> try to execute in parallel
            if "n_jobs" in kwargs.keys():
                n_jobs = kwargs["n_jobs"]
                kwargs.pop("n_jobs")
                if parallel:
                    if n_jobs < 0 or n_jobs > 1:
                        return f_multi(*args, **kwargs)
            return f_single(*args, **kwargs)
        return wrapper

    if f: return decorate(f)
    return decorate


@jit
def uint_sizes():
    out = np.zeros(3, dtype = np.uint64)
    out[0] = np.iinfo(np.uint8).max
    out[1] = np.iinfo(np.uint16).max
    out[2] = np.iinfo(np.uint32).max
    return out
uint_size = uint_sizes()
uint_dtypes = (
    np.zeros(1, dtype = np.uint8),
    np.zeros(1, dtype = np.uint16),
    np.zeros(1, dtype = np.uint32),
    np.zeros(1, dtype = np.uint64))

@jit
def int_sizes():
    out = np.zeros(3, dtype = np.int64)
    out[0] = np.iinfo(np.int8).max
    out[1] = np.iinfo(np.int16).max
    out[2] = np.iinfo(np.int32).max
    return out
int_size = int_sizes()
int_dtypes = (
    np.zeros(1, dtype = np.int8),
    np.zeros(1, dtype = np.int16),
    np.zeros(1, dtype = np.int32),
    np.zeros(1, dtype = np.int64))

###
@jit
def _uinteger(i):
    for j in range(3):
        if i <= uint_size[j]:
            break
    return j

###
@jit
def _integer(i):
    for j in range(3):
        if i <= int_size[j]:
            break
    return j

###
def uinteger(i):
    return uint_dtypes[_uinteger(i)]

###
def integer(i):
    return int_dtypes[_integer(i)]

###
def isComplex(dtype):
    if dtype == np.complex64:
        return True
    elif dtype == np.complex128:
        return True
    elif dtype == complex:
        return True
    return False


@jit
def svd(X, full_matrices = False): 
    return linalg.svd(X, full_matrices = full_matrices)

@jit
def pinv(X): return linalg.pinv(X)

@jit
def eigh(X): return linalg.eigh(X)

# @jit
# def cholesky(X): return linalg.cholesky(X)

@jit
def lstsq(X, y): return linalg.lstsq(X, y.astype(X.dtype))[0]

@jit
def qr(X): return linalg.qr(X)

@jit
def norm(v, d = 2): return linalg.norm(v, d)

@jit
def _0mean(X, axis = 0): return np.sum(X, axis)/X.shape[axis]

def mean(X, axis = 0):
    if axis == 0 and X.flags['C_CONTIGUOUS']:
        return _0mean(X)
    else:
        return X.mean(axis)

@jit
def __sign(X): return np.sign(X)

def sign(X):
    S = __sign(X)
    if isComplex(X.dtype):
        return S.real
    return S

@jit
def _sum(X, axis = 0): return np.sum(X, axis)

@jit
def _abs(v): return np.abs(v)

@jit
def maximum(X, i): return np.maximum(X, i)

@jit
def minimum(X, i): return np.minimum(X, i)

@jit
def _min(a,b):
    if a < b: return a
    return b

@jit
def _max(a,b):
    if a < b: return b
    return a

@jit
def _sign(x):
    if x < 0: return -1
    return 1

@jit
def _arange(size, dtype):
    out = np.zeros(size, dtype = dtype.dtype)
    for i in range(size):
        out[i] = i
    return out

def arange(size):
    return _arange(size, uinteger(size))
    
###
# Run all algorithms to allow caching

# y32 = np.ones(2, dtype = np.float32)
# y64 = np.ones(2, dtype = np.float64)


# X = np.eye(2, dtype = np.float32)
# A = svd(X)
# A = svd(X, False)
# A = eigh(X)
# # A = cholesky(X)
# A = pinv(X)
# A = lstsq(X, y32)
# A = lstsq(X, y64)
# A = qr(X)
# A = norm(y32)
# A = norm(y64)
# A = mean(X)
# A = mean(y32)
# A = mean(y64)
# A = _sum(X)
# A = _sum(y32)
# A = _sum(y64)
# A = sign(X)
# A = sign(y32)
# A = sign(y64)
# A = _abs(y32)
# A = _abs(y64)
# A = _abs(X)
# A = _abs(10.0)
# A = _abs(10)
# A = minimum(X, 0)
# A = minimum(y32, 0)
# A = minimum(y64, 0)
# A = maximum(X, 0)
# A = maximum(y32, 0)
# A = maximum(y64, 0)
# A = _min(0,1)
# A = _min(0.1,1.1)
# A = _max(0,1)
# A = _max(0.1,1.1)
# A = _sign(-1)
# A = _sign(-1.2)
# A = _sign(1.2)

# X = np.eye(2, dtype = np.float64)
# A = svd(X)
# A = svd(X, False)
# A = eigh(X)
# # A = cholesky(X)
# A = pinv(X)
# A = lstsq(X, y32)
# A = lstsq(X, y64)
# A = qr(X)
# A = norm(y32, 2)
# A = norm(y64, 2)
# A = mean(X, 1)
# A = _sum(X)
# A = sign(X)
# A = _abs(X)
# A = maximum(X, 0)
# A = minimum(X, 0)

# del X, A, y32, y64
# A = None
# X = None
# y32 = None
# y64 = None
