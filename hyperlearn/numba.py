
from functools import wraps
from numba import njit as NJIT, prange
import numpy as np

###
FLOAT32_EPS = np.finfo(np.float32).eps
FLOAT64_EPS = np.finfo(np.float64).eps

###
UINT_SIZE = (
    np.iinfo(np.uint8).max,
    np.iinfo(np.uint16).max,
    np.iinfo(np.uint32).max
    )
UINT_DTYPES = (
    np.zeros(1, dtype = np.uint8),
    np.zeros(1, dtype = np.uint16),
    np.zeros(1, dtype = np.uint32),
    np.zeros(1, dtype = np.uint64)
    )

###
INT_SIZE = (
    np.iinfo(np.int8).max,
    np.iinfo(np.int16).max,
    np.iinfo(np.int32).max
    )
INT_DTYPES = (
    np.zeros(1, dtype = np.int8),
    np.zeros(1, dtype = np.int16),
    np.zeros(1, dtype = np.int32),
    np.zeros(1, dtype = np.int64)
    )

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
            f_multi = NJIT(f, fastmath = True, nogil = True, parallel = True, cache = False)
        f_single = NJIT(f, fastmath = True, nogil = True, parallel = False, cache = True)
        
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

###
def njit(f):
    """
    Faster version of @jit. No parallel. Saves a few microseconds.
    """
    F = NJIT(f, fastmath = True, nogil = True, cache = True)
    def wrapper(*args, **kwargs):
        return F(*args, **kwargs)
    return wrapper

###
def fjit(f):
    """
    Fastest version of @jit. No parallel and GIL is NOT released.
    Saves a few more microseconds.
    """
    F = NJIT(f, fastmath = True, cache = True)
    def wrapper(*args, **kwargs):
        return F(*args, **kwargs)
    return wrapper

###
def uinteger(i):
    for j in range(3):
        if i <= UINT_SIZE[j]:
            break
    return UINT_DTYPES[j]

###
def integer(i):
    for j in range(3):
        if i <= INT_SIZE[j]:
            break
    return INT_DTYPES[j]

###
def isComplex(dtype):
    if dtype == np.complex64:
        return True
    elif dtype == np.complex128:
        return True
    elif dtype == complex:
        return True
    return False

###
@njit
def svd(X, full_matrices = False): 
    return np.linalg.svd(X, full_matrices = full_matrices)

###
@njit
def pinv(X): return np.linalg.pinv(X)

###
@njit
def eigh(X): return np.linalg.eigh(X)

# @jit
# def cholesky(X): return np.linalg.cholesky(X)

###
@njit
def lstsq(X, y): return np.linalg.lstsq(X, y.astype(X.dtype))[0]

###
@njit
def qr(X): return np.linalg.qr(X)

###
@njit
def norm(v, d = 2): return np.linalg.norm(v, d)

###
@njit
def __sign(X): return np.sign(X)

###
def sign(X):
    S = __sign(X)
    if isComplex(X.dtype):
        return S.real
    return S

###
@njit
def _sum(X, axis = 0): return np.sum(X, axis)

###
@njit
def maximum(X, i): return np.maximum(X, i)

###
@njit
def minimum(X, i): return np.minimum(X, i)

###
def _min(a,b):
    if a < b: return a
    return b

###
def _max(a,b):
    if a < b: return b
    return a

###
def _sign(x):
    if x < 0: return -1
    return 1

###
def arange(size):
    return np.arange(size, dtype = uinteger(size))


######################################################
# Custom statistical functions
# Mean, Variance
######################################################

###
@NJIT(fastmath = True, nogil = True, cache = True)
def mean_1(X):
    n, p = X.shape
    out = np.zeros(n, dtype = X.dtype)
    for i in range(n):
        s = 0
        for j in range(p):
            s += X[i, j]
        s /= p
        out[i] = s
    return out

###
@NJIT(fastmath = True, nogil = True, cache = True)
def mean_0(X):
    n, p = X.shape
    out = np.zeros(p, dtype = X.dtype)
    for i in range(n):
        for j in range(p):
            out[j] += X[i, j]
    out /= n
    return out

###
@NJIT(fastmath = True, nogil = True, cache = True)
def mean_A(X):
    n, p = X.shape
    s = np.sum(X) / (n*p)
    return 


###
def mean(X, axis = None):
    if axis == 0:
        return mean_0(X)
    elif axis == 1:
        return mean_1(X)
    return mean_A(X)


###
@NJIT(fastmath = True, nogil = True, cache = True)
def var_0(X):
    mu = mean_0(X)
    n, p = X.shape
    variance = np.zeros(p, dtype = mu.dtype)

    for i in range(n):
        for j in range(p):
            v = X[i, j] - mu[j]
            v *= v
            variance[j] += v
    variance /= n-1     # unbiased estimator
    return variance

###
@NJIT(fastmath = True, nogil = True, cache = True)
def var_1(X):
    mu = mean_1(X)
    n, p = X.shape
    variance = np.zeros(n, dtype = mu.dtype)

    for i in range(n):
        _mu = mu[i]
        var = 0
        for j in range(p):
            v = X[i, j] - _mu
            v *= v
            var += v
        variance[i] = var
    variance /= p-1     # unbiased estimator
    return variance

###
@NJIT(fastmath = True, nogil = True, cache = True)
def var_A(X):
    mu = mean_A(X)
    n, p = X.shape

    var = 0
    for i in range(n):
        for j in range(p):
            v = X[i, j] - mu
            v *= v
            var += v
    var /= n*p-1        # unbiased estimator
    return var

###
def var(X, axis = None):
    if axis == 0:
        return var_0(X)
    elif axis == 1:
        return var_1(X)
    return var_A(X)

###
def std(X, axis = None):
    if axis == 0:
        V = var_0(X)
    elif axis == 1:
        V = var_1(X)
    else:
        V = var_A(X)
    return V**0.5

