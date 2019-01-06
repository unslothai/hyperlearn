
import numpy as np
from ..numba.types import *
from ..numba.funcs import arange
from ..random import shuffle, randbool, randint


# Sketching methods from David Woodruff.

###
def sketch(n, p, k = 10, method = "left"):
    """
    Produces a CountSketch matrix which is similar in nature to
    the Johnson–Lindenstrauss Transform (eps Fast-JLT) as shown
    in Sklearn. But, as shown by David Woodruff, "Sketching as a 
    Tool for Numerical Linear Algebra" [arxiv.org/abs/1411.4357],
    super fast matrix multiplies can be done. Notice a difference
    is HyperLearn's  K = min(n, p)/2, but david says k^2/eps is
    needed (too memory taxing). [Added 4/12/18] [Edited 14/12/18]

    Parameters
    -----------
    X:              Data matrix.
    k:              (auto, int). Auto is min(n, p) / 2
    method:         (left, right). Left is S @ X. Right X @ S.
    
    Returns
    -----------
    (Sketch Matrix S or indices)
    """
    if method == "left":
        x = arange(n)
        shuffle(x)
        return x
    else:
        if k < 20:
            sign = randbool(p)
            position = randint(0, k, size = p)
            return _sketch_right(k, n, p, sign, position)
        else:
            x = randint(0, k, size = p)
            return x


###
@jit(parallel = True)
def sketch_multiply_left(X, S, k = 10):
    """
    Multiplies sketch matrix S onto X giving S @ X.
    Very fast and uses little to no memory.
    [Added 14/12/18]
    """
    n, p = X.shape
    size = n//k
        
    out = np.zeros((k, p), dtype = X.dtype)
    partial = np.zeros(p, dtype = X.dtype)

    for i in prange(k-1):
        res = partial.copy()
        left = i*size
        right = (i+1)*size
        middle = int( (i+0.5)*size )

        for j in range(left, middle):
            res += X[S[j]]
        for j in range(middle, right):
            res -= X[S[j]]
        out[i] = res
    
    i = k-1
    left = i*size
    middle = int( (i+0.5)*size )

    for j in range(left, middle):
        partial += X[S[j]]
    for j in range(middle, n):
        partial -= X[S[j]]
    out[i] = partial
    
    return out


###
@jit(parallel = True)
def sketch_multiply_right(X, S, k = 10):
    """
    Multiplies sketch matrix S onto X giving X @ S.
    Can be slow if k is small.
    [Added 14/12/18]
    """
    n, p = X.shape
    out = np.zeros((n, k), dtype = X.dtype)
    
    for i in prange(n):
        Xi = X[i]
        if i % 2 == 0:
            for j in range(p):
                out[i, S[j]] += Xi[j]
        else:
            for b in range(p):
                out[i, S[j]] -= Xi[j]
    return out


###
@jit
def _sketch_right(k, n, p, sign, position):
    S = np.zeros((p, k), dtype = np.int8)
    ones = np.ones(p, dtype = np.int8)
    ones[sign] = -1

    for i in range(p):
        S[i, position[i]] = ones[i]
    return S


###
def sketch_multiply(X, S = None, k = 10, method = "left", n_jobs = 1):
    """
    Multiplies a sketch matrix S onto X either giving SX or XS.
    Tries to be fast and complexity is O(np).
    """
    if S is None:
        S = sketch(X, k, method)
    if method == "left":
        return sketch_multiply_left(X, S, k, n_jobs = n_jobs)
    else:
        if k < 20:
            return X @ S
        else:
            return sketch_multiply_right(X, S, k, n_jobs = n_jobs)

