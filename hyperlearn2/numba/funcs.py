
from .types import *
import numpy as np
from ..cython.utils import uinteger

###
@jit([Tuple((M_32, A32, M_32))(M32_, bool_),Tuple((M_64, A64, M_64))(M64_, bool_),
      Tuple((M_32, A32, M_32))(M_32, bool_),Tuple((M_64, A64, M_64))(M_64, bool_)], **nogil)
def svd(X, full_matrices = False): 
    return np.linalg.svd(X, full_matrices = full_matrices)

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
@jit(**nogil)
def maximum(X, i): return np.maximum(X, i)

###
@jit(**nogil)
def minimum(X, i): return np.minimum(X, i)

###
def arange(size):
    return np.arange(size, dtype = uinteger(size))

###
@jit(M_32(M32_, I64, I64), **parallel)
def asfortranarray_float(X, n, p):
    out = np.empty((p,n), dtype = np.float32)
    for i in prange(n):
        for j in range(p):
            out[j,i] = X[i,j]
    return out.T

###
@jit(M_64(M64_, I64, I64), **parallel)
def asfortranarray_double(X, n, p):
    out = np.empty((p,n), dtype = np.float64)
    for i in prange(n):
        for j in range(p):
            out[j,i] = X[i,j]
    return out.T


###
@jit(M_32(M32_, I64), **nogil)
def copy_symmetric_float(X, n):
    out = np.empty((n,n), dtype = np.float32)

    for i in range(n):
        for j in range(i+1):
            out[i,j] = X[i,j]
    return out.T

###
@jit(M_64(M64_, I64), **nogil)
def copy_symmetric_double(X, n):
    out = np.empty((n,n), dtype = np.float64)

    for i in range(n):
        for j in range(i+1):
            out[i,j] = X[i,j]
    return out.T


###
@jit(M_32(M32_, I64), **parallel)
def copy_symmetric_float_parallel(X, n):
    out = np.empty((n,n), dtype = np.float32)

    for i in prange(n):
        for j in range(i+1):
            out[i,j] = X[i,j]
    return out.T

###
@jit(M_64(M64_, I64), **parallel)
def copy_symmetric_double_parallel(X, n):
    out = np.empty((n,n), dtype = np.float64)

    for i in prange(n):
        for j in range(i+1):
            out[i,j] = X[i,j]
    return out.T

######################################################
# Custom statistical functions
# Mean, Variance
######################################################

###
@jit([A32(M32_), A64(M64_)], **nogil)
def mean_1(X):
    n, p = X.shape
    out = np.empty(n, dtype = X.dtype)
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
    out = np.empty(p, dtype = X.dtype)
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
    variance = np.empty(n, dtype = mu.dtype)

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

