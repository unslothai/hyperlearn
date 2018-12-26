
import numpy as np
from .numba import *
from .base import *
from array import array

###
def epsilon(X):
    if X.itemsize < 8:
        eps = 1000
        eps *= FLOAT32_EPS
    else:
        eps = 1000000
        eps *= FLOAT64_EPS
    return eps

###
def do_until_success(
    f, epsilon_f, size, overwrite = False, alpha = None, *args, **kwargs):
    """
    Epsilon Jitter Algorithm from Modern Big Data Algorithms. Forces certain
    algorithms to converge via ridge regularization.
    [Added 15/11/18] [Edited 25/11/18 Fixed Alpha setting, can now run in
    approx 1 - 2 runs] [Edited 26/11/18 99% rounded accuracy in 1 run]
    [Edited 14/12/18 Some overwrite errors] [Edited 21/12/18 If see a 0]

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

    # Default alpha
    small = X.itemsize < 8
    default = ALPHA_DEFAULT32 if small else ALPHA_DEFAULT64
    
    if alpha is None:
        alpha = 0

    if overwrite:
        alpha = _max(alpha, default)

    if not overwrite:
        previous = X.diagonal().copy()

    old_alpha = 0
    error = 1
    while error != 0:
        epsilon_f(X, size, alpha - old_alpha, default)
        #print(X.diagonal())
        try:
            out = f(*args, **kwargs)
            if isList(out):
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

            if alpha == 0:
                alpha = ALPHA_DEFAULT32 if small else ALPHA_DEFAULT64
            
            print(f"Epsilon Jitter Algorithm Restart with alpha = {alpha}.")
            if overwrite:
                print("Overwriting maybe have issues, please turn it off.")
                overwrite = False

    if not overwrite:
        X.flat[::X.shape[0]+1] = previous
        #epsilon_f(X, size, -alpha)
    return out


###
@fjit
def add_jitter(X, size, alpha, default):
    """
    Epsilon Jitter Algorithm from Modern Big Data Algorithms. Forces certain
    algorithms to converge via ridge regularization.
    [Added 15/11/18] [Edited 25/11/18 for floating point errors]
    [Edited 21/12/18 first pass checks what should be added]
    """
    jitter = alpha

    for i in range(size):
        old = X[i, i]
        if old == 0 and jitter == 0:
            jitter = default

        new = old + jitter
        while new - old < alpha:
            jitter *= jitter
            new = old + jitter            

    for i in range(size):
        X[i, i] += jitter


###
@njit  # Output only triu part. Overwrites matrix to save memory.
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
@njit  # Output only L part. Overwrites LU matrix to save memory.
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


@njit # Force triangular matrix U to be invertible using ridge regularization
def U_process(A, size, alpha, default):
    jitter = alpha

    for i in range(size):
        old = A[i, i]
        if old == 0:
            if jitter == 0:
                jitter = default

            new = jitter
            while old < alpha:
                jitter *= jitter
                new = jitter

    for i in range(size):
        if A[i, i] == 0:
            A[i, i] += jitter


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
def amax_0(X, n, p):
    """
    Finds the sign(X[:,abs(X).argmax(0)])
    """
    indices = np.ones(p, dtype = np.int8)
    maximum = np.zeros(p, dtype = X.dtype)

    for i in prange(n):
        for j in range(p):
            Xij = X[i, j]
            a = abs(Xij)
            if a > maximum[j]:
                maximum[j] = a
                indices[j] = np.sign(Xij)
    return indices


###
@jit(parallel = True)
def amax_1(X, n, p):
    """
    Finds the sign(X[abs(X).argmax(1)])
    """
    indices = np.zeros(n, dtype = np.int8)
    for i in prange(n):
        _max = 0
        s = 1
        for j in range(p):
            Xij = X[i, j]
            a = abs(Xij)
            if a > _max:
                _max = a
                s = np.sign(Xij)
        indices[i] = s
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
    amax = eval(f"amax_{axis}")

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
@fjit
def svd_search(S, eps, size):
    """
    Determines the rank of a matrix via the singular values.
    (Counts how many are larger than eps.)
    """
    rank = size-1
    if S[rank] < eps:
        rank -= 1
        while rank > 0:
            if S[rank] >= eps: break
            rank -= 1
    rank += 1
    return rank


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
    eps = epsilon(S)*S[0]

    # Binary search O(logn) is not useful
    # since most singular values are not going to be 0
    size = S.size
    rank = svd_search(S, eps, size)
    if rank != size:
        U, S, VT = U[:, :rank], S[:rank], VT[:rank]

    # Check if alpha needs to be added onto the singular values
    alphaUpdate = False
    if alpha is not None:
        if is32:
            if alpha != ALPHA_DEFAULT32:
                alphaUpdate = True
        else:
            if alpha != ALPHA_DEFAULT64:
                alphaUpdate = True

    if alphaUpdate:
        S /= (S**2 + alpha)
    else:
        S = 1/S

    return U, S, VT


###
@fjit
def eig_search(W, eps):
    """
    Corrects the eigenvalues if they're smaller than 0.
    """
    for i in range(W.size):
        if W[i] > 0: break
        # else set to condition number
        W[i] = eps
    return W


###
def eig_condition(X, W, V):
    # Condition number just in case W[i] <= 0
    eps = epsilon(W)
    if W[-1] >= 0:
        first = W[-1]**0.5 # eigenvalues are sorted ascending
    else:
        first = 0
    eps *= first
    eps **= 2 # since eigenvalues are squared of singular values

    W = eig_search(W, eps)    

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


@fjit
def eigh_search(W, eps):
    w0 = abs(W[0])

    last = W[-1]
    negative = (last < 0)
    w1 = abs(last)

    cutoff = w0 if w0 > w1 else w1
    cutoff *= eps
    
    size = len(W)
    out = np.ones(size, dtype = np.bool_)
    
    if w0 < cutoff:
        out[0] = False
    if w1 < cutoff:
        out[-1] = False
    
    # Check if abs(W) < eps
    for i in range(1, size-1):
        if abs(W[i]) < cutoff:
            out[i] = False
            
    return out


###
@njit
def _row_norm(X):
    n, p = X.shape
    norm = np.zeros(n, dtype = X.dtype)
    
    for i in range(n):
        s = 0
        for j in range(p):
            xij = X[i, j]
            xij *= xij
            s += xij
        norm[i] = s
    return norm

###
@njit
def _col_norm(X):
    n, p = X.shape
    norm = np.zeros(p, dtype = X.dtype)
    
    for i in range(n):
        for j in range(p):
            xij = X[i, j]
            xij *= xij
            norm[j] += xij
    return norm

###
@fjit
def normA(X):
    s = 0
    for i in range(X.size):
        xi = X[i]
        xi *= xi
        s += xi
    return s

###
def col_norm(X):
    if len(X.shape) > 1:
        return _col_norm(X)
    return normA(X)

###
def row_norm(X):
    if len(X.shape) > 1:
        return _row_norm(X)
    return normA(X)


###
@fjit
def _frobenius_norm(X):
    n, p = X.shape
    norm = 0
    
    for i in range(n):
        for j in range(p):
            xij = X[i, j]
            xij *= xij
            norm += xij
    return norm

###
@fjit
def _frobenius_norm_symmetric(X):
    n = X.shape[0]
    norm = 0
    diag = 0
    
    for i in range(n):
        for j in range(i+1, n):
            xij = X[i, j]
            xij *= xij
            norm += xij

        xii = X[i, i]
        xii *= xii
        diag += xii
    norm *= 2
    norm += diag
    return norm

###
def frobenius_norm(X, symmetric = False):
    """
    Outputs the ||X||_F ^ 2 norm of a matrix. If symmetric,
    then computes the norm on 1/2 of the matrix and multiplies
    it by 2.
    """
    if len(X.shape) > 1:
        if symmetric:
            return _frobenius_norm_symmetric(X)
        return _frobenius_norm(X)
    return normA(X)


###
@njit
def proportion(X):
    X /= np.sum(X)
    return X


###
@njit
def gram_schmidt(X, P, n, k):
    """
    Modified stable Gram Schmidt process.
    Gram-Schmidt Orthogonalization
    Instructor: Ana Rita Pires (MIT 18.06SC).
    Output is Q.T NOT Q. So you must transpose it.
    """
    Q = np.zeros((k, n), dtype = X.dtype)
    Z = np.zeros(n, dtype = X.dtype)

    for i in range(k):

        x = Q[i]
        col = P[i]
        # Q[i] = X[:,P[i]]
        for a in range(n):
            x[a] = X[a, col]
        
        for j in range(i):
            q = Q[j]
            x -= np.vdot(x, q) * q
            
        norm = np.linalg.norm(x)
        if norm == 0:
            Q[i] = Z
            Q[i,i] = 1
        else:
            x /= norm
    return Q


###
@njit
def _unique_int(a):
    """
    Assumes a is just integers, and returning unique elements
    will be much easier. Uses a quick boolean array instead of
    a hash table.
    """
    seen = np.zeros(np.max(a)+1, dtype = np.bool_)
    count = 0
    
    for i in range(a.size):
        element = a[i]
        curr = seen[element]
        if not curr:
            seen[element] = True
            count += 1

    out = np.zeros(count, dtype = a.dtype)
    j = 0
    # fill up array with uniques
    for i in range(seen.size):
        if seen[i]:
            out[j] = i
            j += 1
            if j > count: break
    return out

###
@njit
def _unique_count(a, size):
    """
    Returns the counts and unique values of an array a.
    [Added 23/12/18]
    """
    maximum = np.max(a) + 1
    seen = np.zeros(maximum, dtype = size.dtype)
    count = 0
    
    for i in range(a.size):
        element = a[i]
        curr = seen[element]
        if curr == 0: count += 1
        seen[element] += 1

    unique = np.zeros(count, dtype = a.dtype)
    counts = np.zeros(count, dtype = np.uint32)
    
    j = 0
    for i in range(seen.size):
        curr = seen[i]
        if curr > 0:
            unique[j] = i
            counts[j] = curr
            j += 1
            if j > count: break
    return unique, counts

###
@NJIT(fastmath = True, cache = True)
def _unique_sorted_size(a):
    """
    Returns how many uniques in a sorted list.
    [Added 23/12/18]
    """
    size = 1
    i = 0
    old = a[i]
    i += 1
    while i < a.size:
        new = a[i]
        if new != old:
            size += 1
            old = new
        i += 1
    return size

###
@fjit
def _unique_sorted(a):
    """
    Returns only unique elements in a sorted list.
    [Added 23/12/18]
    """
    size = _unique_sorted_size(a)
        
    out = np.zeros(size, dtype = a.dtype)
    i = 0
    old = a[i]
    out[i] = old
    
    i += 1
    j = 1
    length = a.size
    while i < length:
        new = a[i]
        if new != old:
            out[j] = new
            old = new
            j += 1
            if j > length: break
        i += 1
    return out

###
@fjit
def _unique_sorted_count(a):
    """
    Returns unique elements and their counts in a sorted list.
    [Added 23/12/18]
    """
    size = _unique_sorted_size(a)
        
    out = np.zeros(size, dtype = a.dtype)
    counts = np.zeros(size, dtype = np.uint32)
    i = 0
    old = a[i]
    out[i] = old
    
    i += 1
    j = 0
    length = a.size
    while i < length:
        new = a[i]

        # Add 1 to count
        counts[j] += 1
        if new != old:
            j += 1
            if j > length: break
            out[j] = new
            old = new
        i += 1

    # Need to update last element since loop forgets it.
    counts[-1] += 1
    return out, counts


###
def unique_int(a, return_counts = False):
    """
    Given a list of postive ints, returns the unique
    elements accompanied with optional counts.
    [Added 23/12/18]

    Parameters
    -----------
    a:              Array of postive ints
    return_counts:  Whether to return (unique, counts)
    """
    if return_counts:
        return _unique_count(a, uinteger(a.size))
    return _unique_int(a)


###
def unique_sorted(a, return_counts = False):
    """
    Given a list of sorted elements, returns the unique
    elements accompanied with optional counts.
    [Added 23/12/18]

    Parameters
    -----------
    a:              Array of sorted elements
    return_counts:  Whether to return (unique, counts)
    """
    if return_counts:
        return _unique_sorted_count(a)
    return _unique_sorted(a)

