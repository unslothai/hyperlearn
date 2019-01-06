
from .types import *
import numpy as np
from ..cfuncs import uinteger

###
@jit([Tuple((M_32, A32, M_32))(M_32, bool_),Tuple((M_64, A64, M_64))(M64_, bool_),
      Tuple((M_32, A32, M_32))(M_32, bool_),Tuple((M_64, A64, M_64))(M_64, bool_)], **nogil)
def svd(X, full_matrices = False): 
    return np.linalg.svd(X, full_matrices = full_matrices)

###
# @jit([M32(M32), M64(M64)], **nogil)
# def pinv(X): return np.linalg.pinv(X)

###
# @jit([(A32, M32)(M32), (A64, M64)(M64)], **nogil)
# def eigh(X): return np.linalg.eigh(X)

# @jit
# def cholesky(X): return np.linalg.cholesky(X)

###
@jit([A32(M32_, A32), A64(M64_, A64)], **nogil)
def lstsq(X, y): return np.linalg.lstsq(X, y.astype(X.dtype))[0]

###
@jit([Tuple((M32, M_32))(M32_), Tuple((M64, M_64))(M64_)], **nogil)
def qr(X): return np.linalg.qr(X)

###
@jit([F64(A64), F64(A32)])
def norm(v): return np.linalg.norm(v)

###
# @njit
# def __sign(X): return np.sign(X)

###
# def sign(X):
#     S = __sign(X)
#     if isComplex(X.dtype):
#         return S.real
#     return S

###
# @njit
# def _sum(X, axis = 0): return np.sum(X, axis)

###
@jit(**nogil)
def maximum(X, i): return np.maximum(X, i)

###
@jit(**nogil)
def minimum(X, i): return np.minimum(X, i)


###
def arange(size):
    return np.arange(size, dtype = uinteger(size))


######################################################
# Custom statistical functions
# Mean, Variance
######################################################

###
@jit([A32(M32_), A64(M64_)], **nogil)
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
@jit([A32(M32_), A64(M64_)], **nogil)
def mean_0(X):
    n, p = X.shape
    out = np.zeros(p, dtype = X.dtype)
    for i in range(n):
        for j in range(p):
            out[j] += X[i, j]
    out /= n
    return out

###
@jit([F64(M32_), F64(M64_)], **nogil)
def mean_A(X):
    n, p = X.shape
    s = np.sum(X) / (n*p)
    return s


###
def mean(X, axis = None):
    if axis == 0:
        return mean_0(X)
    elif axis == 1:
        return mean_1(X)
    return mean_A(X)


###
@jit([A32(M32_), A64(M64_)], **nogil)
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
@jit([A32(M32_), A64(M64_)], **nogil)
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
@jit([F64(M32_), F64(M64_)], **nogil)
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

