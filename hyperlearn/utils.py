
import numpy as np
from .numba import jit, prange, sign, _max
from .base import *
dtypes = (np.uint8, np.uint16, np.uint32, np.uint64)
from array import array

###
def uint(i, array = True):
    """
    [Added 14/11/2018]
    Outputs a small array with the correct dtype that minimises
    memory usage for storing the maximal number of elements of
    size i

    input:      1 argument
    ----------------------------------------------------------
    i:          Size of maximal elements you want to store

    returns:    array of size 1 with correct dtype
    ----------------------------------------------------------
    """
    for x in dtypes:
        if np.iinfo(x).max >= i:
            break
    if array:
        return np.empty(1, dtype = x)
    return x


###
def do_until_success(f, epsilon_f, size, overwrite = False, alpha = None, *args, **kwargs):
    """
    Epsilon Jitter Algorithm from Modern Big Data Algorithms. Forces certain
    algorithms to converge via ridge regularization.
    [Added 15/11/18] [Edited 25/11/18 Fixed Alpha setting, can now run in
    approx 1 - 2 runs] [Edited 26/11/18 99% rounded accuracy in 1 run]

    Parameters
    -----------
    f:          Function for solver
    epsilon_f:  How to add epsilon
    size:       Argument for epsilon_f
    overwrite:  Whether to overwrite data matrix
    alpha:      Ridge regularizer - default = 1e-6
    """
    if len(args) > 0:
        X = args[0]
    else:
        X = next(iter(kwargs.values()))
    
    if alpha is None:
        alpha = 0
    alpha = _max(alpha, ALPHA_DEFAULT32 if X.itemsize < 8 else ALPHA_DEFAULT64)
    if overwrite:
        alpha *= 10

    old_alpha = 0
    error = 1
    while error != 0:
        epsilon_f(X, size, alpha - old_alpha)
        try:
            out = f(*args, **kwargs)
            if type(out) == tuple:
                error = out[-1]
                out = out[:-1]
                if len(out) == 1:
                    out = out[0]
            else:
                error = 0
        except: 
            pass
        if error != 0:
            old_alpha = alpha
            alpha *= 2
            print(f"Epsilon Jitter Algorithm Restart with alpha = {alpha}.")
            if overwrite:
                print("Overwriting maybe have issues, please turn it off.")
                overwrite = False

    if not overwrite:
        epsilon_f(X, size, -alpha)
    return out


###
@jit
def add_jitter(X, size, alpha):
    """
    Epsilon Jitter Algorithm from Modern Big Data Algorithms. Forces certain
    algorithms to converge via ridge regularization.
    [Added 15/11/18] [Edited 25/11/2018 for floating point errors]
    """
    for i in range(size):
        old = X[i, i]
        X[i, i] += alpha
        multiplier = 2
        while X[i, i] - old < alpha:
            X[i, i] += multiplier*alpha
            multiplier *= 2


###
@jit  # Output only triu part. Overwrites matrix to save memory.
def triu(n, p, U):
    tall = (n > p)
    k = n

    if tall:
        # tall matrix
        # U get all p rows
        U = U[:p]
        k = p

    for i in range(1, k):
        U[i, :i] = 0
    return U


###
@jit  # Output only L part. Overwrites LU matrix to save memory.
def L_process(n, p, L):
    if p > n:
        # wide matrix
        # L get all n rows, but only n columns
        L = L[:, :n]
        k = n
    else:
        k = p

    # tall / wide matrix
    for i in range(k):
        li = L[i]
        li[i+1:] = 0
        li[i] = 1
    # Set diagonal to 1
    return L, k


@jit # Force triangular matrix U to be invertible using ridge regularization
def U_process(A, size, alpha):
    for i in range(size):
        if A[i, i] == 0:
            A[i, i] += alpha
            multiplier = 2
            if A[i, i] < alpha:
                A[i, i] += multiplier*alpha
                multiplier *= 2


###
@jit(parallel = True)
def _reflect(X):
    n = X.shape[0]
    for i in prange(1, n):
        Xi = X[i]
        for j in range(i):
            X[j, i] = Xi[j]
    return X

###
def reflect(X):
    n = X.shape[0]
    if n > 5000:
        n_jobs = -1
    else:
        n_jobs = 1

    if X.flags["F_CONTIGUOUS"]:
        return _reflect(X.T, n_jobs = n_jobs)
    else:
        return _reflect(X, n_jobs = n_jobs)

###
@jit(parallel = True)        
def amax_numb_0(X, n, p):
    indices = np.zeros(p, dtype = np.dtype('int'))
    for i in prange(p):
        _max = 0
        s = 1
        for j in range(n):
            Xji = X[j, i]
            a = abs(Xji)
            if a > _max:
                _max = a
                s = np.sign(Xji)
        indices[i] = s
    return indices

###
@jit(parallel = True)
def amax_numb_1(X, n, p):
    indices = np.zeros(n, dtype = np.dtype('int'))
    for i in prange(n):
        _max = 0
        s = 1
        Xi = X[i]
        for j in range(p):
            Xij = Xi[j]
            a = abs(Xij)
            if a > _max:
                _max = a
                s = np.sign(Xij)
        indices[i] = s
    return indices

###
@jit(parallel = True)        
def amax_numb_0c(X, n, p):
    indices = np.zeros(p, dtype = np.dtype('int'))
    for i in prange(p):
        _max = 0
        s = 1
        for j in range(n):
            Xji = X[j, i]
            a = abs(Xji)
            if a > _max:
                _max = a
                s = np.sign(Xji)
        indices[i] = s.real
    return indices

###
@jit(parallel = True)
def amax_numb_1c(X, n, p):
    indices = np.zeros(n, dtype = np.dtype('int'))
    for i in prange(n):
        _max = 0
        s = 1
        Xi = X[i]
        for j in range(p):
            Xij = Xi[j]
            a = abs(Xij)
            if a > _max:
                _max = a
                s = np.sign(Xij)
        indices[i] = s.real
    return indices

###
def sign_max(X, axis = 0, n_jobs = 1):
    """
    [Added 19/11/2018] [Edited 24/11/2018 Uses NUMBA]
    Returns the sign of the maximum absolute value of an axis of X.

    input:      1 argument, 2 optional
    ----------------------------------------------------------
    X:          Matrix X to be processed. Must be 2D array.
    axis:       Default = 0. 0 = column-wise. 1 = row-wise.
    n_jobs:     Default = 1. Uses multiple CPUs if n*p > 20,000^2.

    returns:    sign array of -1,1
    ----------------------------------------------------------
    """ 
    n, p = X.shape
    if isComplex(X.dtype):
        amax = eval(f"amax_numb_{axis}c")
    else:
        amax = eval(f"amax_numb_{axis}")

    if n_jobs != 1 and n*p > 20000**2:
        return amax(X, n, p, n_jobs = -1)
    else:
        return amax(X, n, p, n_jobs = 1)


###
def svd_flip(U = None, VT = None, U_decision = False, n_jobs = 1):
    """
    [Added 19/11/2018] [Edited 24/11/2018 Uses NUMBA]
    Flips the signs of U and VT from a SVD or eigendecomposition.
    Default opposite to Sklearn's U decision. HyperLearn uses
    the maximum of rows of VT.

    input:      2 argument, 1 optional
    ----------------------------------------------------------
    U:          U matrix from SVD
    VT:         VT matrix (V transpose) from SVD
    U_decision: Default = False. If True, uses max from U.

    returns:    Nothing. Inplace updates U, VT
    ----------------------------------------------------------
    """
    if U_decision is None:
        return

    if U is not None:
        if U_decision:
            signs = sign_max(U, 0, n_jobs = n_jobs)
        else:
            signs = sign_max(VT, 1, n_jobs = n_jobs)
        U *= signs
        VT *= signs[:,np.newaxis]
    else:
        # Eig flip on eigenvectors
        signs = sign_max(VT, 0, n_jobs = n_jobs)
        VT *= signs

###
def svd_condition(U, S, VT, alpha = None):
    """
    [Added 21/11/2018]
    Uses Scipy's SVD condition number calculation to improve pseudoinverse
    stability. Uses (1e3, 1e6) * eps(S) * S[0] as the condition number.
    Everythimg below cond is set to 0.

    input:      3 argument, 1 optional
    ----------------------------------------------------------
    U:          U matrix from SVD
    S:          S diagonal array from SVD
    VT:         VT matrix (V transpose) from SVD
    alpha:      Default = None. Ridge regularization

    returns:    U, S/(S+alpha), VT updated.
    ----------------------------------------------------------
    """
    dtype = S.dtype.char.lower()
    if dtype == "f":
        cond = 1000 #1e3
    else:
        cond = 1000000 #1e6
    eps = np.finfo(dtype).eps
    first = S[0]
    eps = eps*first*cond

    # Binary search O(logn) is not useful
    # since most singular values are not going to be 0
    size = S.size
    rank = size-1
    if S[rank] < eps:
        rank -= 1
        while rank > 0:
            if S[rank] >= eps: break
            rank -= 1
    rank += 1

    if rank != size:
        U, S, VT = U[:, :rank], S[:rank], VT[:rank]

    update = True
    if alpha != 1e-8:
        if alpha != None:
            S /= (S**2 + alpha)
            update = False

    if update:
        S = np.divide(1, S, out = S)

    return U, S, VT


###
def eig_condition(X, W, V):
    # Condition number just in case W[i] <= 0

    dtype = W.dtype.char.lower()
    if dtype == "f":
        cond = 1000 #1e3
    else:
        cond = 1000000 #1e6
    eps = np.finfo(dtype).eps
    first = W[-1]**0.5 # eigenvalues are sorted ascending
    eps = eps*first*cond
    eps **= 2 # since eigenvalues are squared of singular values

    for i in range(W.size):
        if W[i] > 0: break
        # else set to condition number
        W[i] = eps

    # X.H @ V / W**0.5
    XT = X if X.flags["F_CONTIGUOUS"] else X.T
    
    if isComplex(dtype):
        if isComplex(dtypeY):
            out = blas("gemm")(a = XT, b = V.T, trans_a = 0, trans_b = 2, alpha = 1)
        else:
            out = XT @ V
        out = np.conjugate(out, out = out)
    else:
        out = XT @ V

    out /= W**0.5
    return W, out

    
# #### Load JIT
# X = np.eye(2, dtype = np.float32)
# svd_flip(X, X)
# do_until_success(lapack("potrf"), add_jitter, 2, False, None, X)
# del X;
# X = np.eye(2, dtype = np.float64)
# svd_flip(X, X)
# do_until_success(lapack("potrf"), add_jitter, 2, False, None, X)
# del X;

###
@jit
def row_norm(X):
    n, p = X.shape
    norm = np.zeros(n, dtype = X.dtype)
    
    for i in range(n):
        row = X[i]
        s = 0
        for j in range(p):
            s += row[j]**2
        norm[i] = s
    return norm

###
@jit
def col_norm(X):
    n, p = X.shape
    norm = np.zeros(p, dtype = X.dtype)
    
    for i in range(n):
        row = X[i]
        for j in range(p):
            norm[j] += row[j]**2
    return norm

