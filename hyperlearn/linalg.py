
"""
linalg.py
====================================
Linear Algebra Functions for HyperLearn
"""

from .base import *
from .utils import *
import scipy.linalg as scipy
from .numba import jit, _max, _min
from . import numba


###
def transpose(X, overwrite = True, dtype = None):
    """
    Provides X.T if dtype == float, else X.H (Conjugate transpose)
    [Added 23/11/18]

    Parameters
    -----------
    X :         Matrix to be decomposed. Has to be symmetric.
    Overwrite:  If overwritten, then inplace operation.

    Returns
    -----------
    X.T or X.H: Conjugate Tranpose (X)
    """
    if dtype is None:
        dtype = X.dtype
    if isComplex(dtype):
        if overwrite:
            return np.conjugate(X, out = X).T
        return X.conj().T
    return X.T


###
@process(square = True, memcheck = "columns")
def cholesky(X, alpha = None):
    """
    Uses Epsilon Jitter Solver to compute the Cholesky Decomposition
    until success. Default alpha ridge regularization = 1e-6.
    [Added 15/11/18] [Edited 16/11/18 Numpy is slower. Uses LAPACK only]
    [Edited 18/11/18 Uses universal "do_until_success"]

    Parameters
    -----------
    X :         Matrix to be decomposed. Has to be symmetric.
    alpha :     Ridge alpha regularization parameter. Default 1e-6
    turbo :     Boolean to use float32, rather than more accurate float64.

    Returns
    -----------
    U :         Upper triangular cholesky factor (U)
    """
    decomp = lapack("potrf", None)
    size = X.shape[0] + 1 # add items to diagonal
    U = do_until_success(decomp, add_jitter, size, False, alpha, X) # overwrite matters
    return U

###
@process(square = True, memcheck = "columns")
def cho_solve(X, rhs, alpha = None):
    """
    Given U from a cholesky decompostion and a RHS, find a least squares
    solution.
    [Added 15/11/18]

    Parameters
    -----------
    X :         Cholesky Factor. Use cholesky first.
    alpha :     Ridge alpha regularization parameter. Default 1e-6
    turbo :     Boolean to use float32, rather than more accurate float64.

    Returns
    -----------
    U :          Upper triangular cholesky factor (U)
    """
    theta = lapack("potrs")(X, rhs)[0]
    return theta

###
@process(square = True, memcheck = "squared")
def cho_inv(X, turbo = True):
    """
    Computes an inverse to the Cholesky Decomposition.
    [Added 17/11/18]

    Parameters
    -----------
    X :         Upper Triangular Cholesky Factor U. Use cholesky first.
    turbo :     Boolean to use float32, rather than more accurate float64.
    
    Returns
    -----------
    inv(U) :     Upper Triangular Inverse(X)
    """
    inv = lapack("potri", None, turbo)(X)
    return inv

###
@process(memcheck = "full")
def pinvc(X, alpha = None, turbo = True, conjugate = True, overwrite = False):
    """
    Returns the Pseudoinverse of the matrix X using Cholesky Decomposition.
    Fastest pinv(X) possible, and uses the Epsilon Jitter Algorithm to
    guarantee convergence. Allows Ridge Regularization - default 1e-6.
    [Added 17/11/18] [Edited 18/11/18 for speed - uses more BLAS]
    [Edited 23/11/18 Added Complex support]

    Parameters
    -----------
    X :         General matrix X.
    alpha :     Ridge alpha regularization parameter. Default 1e-6
    turbo :     Boolean to use float32, rather than more accurate float64.
    conjugate:  Whether to inplace conjugate but and inplace return original.
    overwrite:  Whether to conjugate transpose inplace.

    Returns
    -----------    
    pinv(X) :   Pseudoinverse of X. Allows pinv(X) @ X = I if n >= p or X
                @ pinv(X) = I for p > n.
    """
    n, p = X.shape
    size = X.shape[0] + 1 # add items to diagonal
    # determine under / over-determined
    XXT = p > n
    dtype = X.dtype
    a = transpose(X, conjugate, dtype)

    # get covariance or gram matrix
    U = blas("syrk")(a = a, trans = XXT, alpha = 1)

    decomp = lapack("potrf", None)
    U = do_until_success(decomp, add_jitter, size, True, alpha, U) # overwrite shouldnt matter
    U = lapack("potri", None, turbo)(U, overwrite_c = True)[0]

    # if XXT -> XT * (XXT)^-1
    # if XTX -> (XTX)^-1 * XT
    inv = blas("symm")(a = U, b = a, side = XXT, alpha = 1)
    if not overwrite and conjugate and isComplex(dtype):
        transpose(X, True, dtype);
    return inv


###
@process(square = True, memcheck = "full")
def pinvh(X, alpha = None, turbo = True, n_jobs = 1):
    """
    Returns the inverse of a square Hermitian Matrix using Cholesky 
    Decomposition. Uses the Epsilon Jitter Algorithm to guarantee convergence. 
    Allows Ridge Regularization - default 1e-6.
    [Added 19/11/18]

    Parameters
    -----------
    X :         Upper Symmetric Matrix X
    alpha :     Ridge alpha regularization parameter. Default 1e-6
    turbo :     Boolean to use float32, rather than more accurate float64.
    n_jobs :    Whether to perform multiprocessing.

    Returns
    -----------    
    pinv(X) :   Pseudoinverse of X. Allows pinv(X) @ X = I.
    """
    decomp = lapack("potrf", None)
    U = do_until_success(decomp, add_jitter, X.shape[0] + 1, False, alpha, U)
    U = lapack("potri", None, turbo)(U, overwrite_c = True)[0]

    return _reflect(U, n_jobs = n_jobs)


###
@process(memcheck = {"X":"full", "L_only":"same", "U_only":"same"})
def lu(X, L_only = False, U_only = False, overwrite = False):
    """
    Computes the pivoted LU decomposition of a matrix. Optional to output
    only L or U components with minimal memory copying.
    [Added 16/11/18]

    Parameters
    -----------
    X:          Matrix to be decomposed. Can be retangular.
    L_only:     Output only L.
    U_only:     Output only U.
    overwrite:  Whether to directly alter the original matrix.

    Returns
    -----------    
    (L,U) or (L) or (U)
    """
    n, p = X.shape
    if L_only or U_only:
        A, P, __ = lapack("getrf")(X, overwrite_a = overwrite)
        if L_only:
            ###
            @jit  # Output only L part. Overwrites LU matrix to save memory.
            def L_process(n, p, L):
                wide = (p > n)
                k = p

                if wide:
                    # wide matrix
                    # L get all n rows, but only n columns
                    L = L[:, :n]
                    k = n

                # tall / wide matrix
                for i in range(k):
                    li = L[i]
                    li[i+1:] = 0
                    li[i] = 1
                # Set diagonal to 1
                return L, k
            A, k = L_process(n, p, A)
            # inc = -1 means reverse order pivoting
            A = lapack("laswp")(a = A, piv = P, inc = -1, k1 = 0, k2 = k-1, overwrite_a = True)
        else:
            # get only upper triangle
            A = triu(n, p, A)
        return A
    else:
        return scipy.lu(X, permute_l = True, check_finite = False, overwrite_a = overwrite)


###
@process(square = True, memcheck = "same")
def pinvl(X, alpha = None, turbo = True, overwrite = False):
    """
    [Added 18/11/18]
    Computes the pseudoinverse of a square matrix X using LU Decomposition.
    Notice, it's much faster to use pinvc (Choleksy Inverse).

    Parameters
    -----------
    X:          Matrix to be decomposed. Must be square.
    alpha:      Ridge alpha regularization parameter. Default 1e-6
    turbo:      Boolean to use float32, rather than more accurate float64.
    overwrite:  Whether to directly alter the original matrix.

    Returns
    -----------    
    pinv(X):    Pseudoinverse of X. Allows pinv(X) @ X = I = X @ pinv(X) 
    """
    n, p = X.shape
    size = n if n < p else p

    A, P, __ = lapack("getrf")(X, overwrite_a = overwrite)

    @jit # Force triangular matrix U to be invertible using ridge regularization
    def U_process(A, size, alpha):
        for i in range(size):
            if A[i, i] == 0:
                A[i, i] += alpha

    inv = lapack("getri")
    A = do_until_success(inv, U_process, size, True, alpha, lu = A, piv = P, overwrite_lu = True) 
    # overwrite shouldnt matter
    return A


###
@process(memcheck = {"X":"full", "Q_only":"same", "R_only":"same"})
def qr(X, Q_only = False, R_only = False, overwrite = False):
    """
    Computes the reduced economic QR Decomposition of a matrix. Optional
    to output only Q or R.
    [Added 16/11/18]

    Parameters
    -----------
    X:          Matrix to be decomposed. Can be retangular.
    Q_only:     Output only Q.
    R_only:     Output only R.
    overwrite:  Whether to directly alter the original matrix.

    Returns
    -----------    
    (Q,R) or (Q) or (R)
    """
    if Q_only or R_only:
        n, p = X.shape
        R, tau, __, __ = lapack("geqrf")(X, overwrite_a = overwrite)

        if Q_only:
            if p > n:
                R = R[:, :n]
            # Compute Q
            Q, __, __ = lapack("orgqr")(R, tau, overwrite_a = True)
            return Q
        else:
            # get only upper triangle
            R = triu(n, p, R)
            return R

    return lapack(None, "qr")(X)


###
def svd_lwork(dtype, byte, n, p):
    """
    Computes the work required for SVD (gesdd, gesvd)
    """
    if n < p:
        MIN = n
        MAX = p
    else:
        MIN = p
        MAX = n

    # check memory usage for GESDD vs GESVD. Use one which is within memory limits.
    if isComplex(dtype):
        gesdd = (MIN + 3)*MIN  # updated from netlib
        gesvd = 2*MIN + MAX
    else:
        # min(n, p)*(6 + min(n,p)) + max(n, p)
        gesdd = (4*MIN + 7)*MIN   # updated from netlib
        gesvd = _max(3*MIN + MAX, 5*MIN)

    gesdd *= byte; gesvd *= byte;
    gesdd >>= 20; gesvd >>= 20;
    return gesdd, gesvd


###
@process(memcheck = "extended")
def svd(X, U_decision = False, n_jobs = 1, conjugate = True, overwrite = False):
    """
    Computes the Singular Value Decomposition of a general matrix providing
    X = U S VT. Notice VT (V transpose) is returned, and not V.
    Also, by default, the signs of U and VT are swapped so that VT has the
    sign of the maximum item as positive.

    HyperLearn's SVD is optimized dramatically due to the findings made in
    Modern Big Data Algorithms. If p/n >= 0.001, then GESDD is used. Else,
    GESVD is used. Also, svd(XT) is used if it's faster, bringing the complexity
    to O( min(np^2, n^2p) ).
    [Added 19/11/18] [Edited 23/11/18 Added Complex support]
    
    Parameters
    -----------
    X:          Matrix to be decomposed. General matrix.
    U_decision: Default = False. If True, uses max from U. If None. don't flip.
    n_jobs:     Whether to use more >= 1 CPU
    conjugate:  Whether to inplace conjugate but and inplace return original.
    overwrite:  Whether to conjugate transpose inplace.

    Returns
    -----------    
    U:          Orthogonal Left Eigenvectors
    S:          Descending Singular Values
    VT:         Orthogonal Right Eigenvectors
    """
    n, p = X.shape
    dtype = X.dtype
    ifTranspose = p > n # p > n
    if ifTranspose: 
        X = transpose(X, conjugate, dtype)
        U_decision = not U_decision
        n, p = X.shape
    byte = X.itemsize

    gesdd, gesvd = svd_lwork(dtype, byte, n, p)
    free = available_memory()
    if gesdd > free:
        if gesvd > free:
            raise MemoryError(f"GESVD requires {gesvd} MB, but {free} MB is free, "
    f"so an extra {gesvd-free} MB is required.")
        gesdd = False
    else:
        gesdd = True

    # Use GESDD from Numba or GESVD from LAPACK
    ratio = p/n
    # From Modern Big Data Algorithms -> GESVD better if matrix is very skinny
    if ratio >= 0.001:
        if overwrite:
            U, S, VT, __ = lapack("gesdd")(X, full_matrices = False, overwrite_a = overwrite)
        else:
            U, S, VT = numba.svd(X, full_matrices = False)
    else:
        U, S, VT, __ = lapack("gesvd")(X, full_matrices = False, overwrite_a  = overwrite)
        
    # Return original X if X.H
    if not overwrite and conjugate and isComplex(dtype):
        transpose(X, True, dtype);

    # In place flips sign according to max(abs(VT))
    svd_flip(U, VT, U_decision = U_decision, n_jobs = n_jobs)
    
    # Flip if svd(X.T) was performed.
    if ifTranspose:
        return transpose(VT, True, dtype), S, transpose(U, True, dtype)
    return U, S, VT


###
@process(memcheck = "extended")
def pinv(X, alpha = None, overwrite = False):
    """
    Returns the inverse of a general Matrix using SVD. Uses the Epsilon Jitter 
    Algorithm to guarantee convergence. Allows Ridge Regularization - default 1e-6.
    [Added 21/11/18] [Edited 23/11/18 Added Complex support]

    Parameters
    -----------
    X:          Upper Triangular Cholesky Factor U. Use cholesky.
    alpha:      Ridge alpha regularization parameter. Default 1e-6.
    overwrite:  Whether to directly alter the original matrix.

    Returns
    -----------    
    pinv(X):    Pseudoinverse of X. Allows pinv(X) @ X = I.
    """
    dtype = X.dtype
    U, S, VT = svd(X, U_decision = None, overwrite = overwrite)
    U, S, VT = svd_condition(U, S, VT, alpha)
    return (transpose(VT, True, dtype) * S) @ transpose(U, True, dtype)


###
def eigh_lwork(dtype, byte, n, p):
    """
    Computes the work required for EIGH (syevr, syevd, heevr, heevd)
    SYEVD = 1 + 6n + 2n^2
    SYEVR = 26n
    HEEVD = 2n + n^2
    HEEVR = 2n
    """
    if p >= 1.1*n:
        s = n
    else:
        s = p

    if isComplex(dtype):
        heevd = 2*s + s**2
        heevr = 2*s + 1
        evd, evr = heevd, heevr
    else:
        syevd = 1 + 6*s + 2*s**2
        syevr = 26*s
        evd, evr = syevd, syevr

    evd *= byte; evr *= byte;
    evd >>= 20; evr >>= 20;
    return evd, evr


###
@process(square = True, memcheck = "extra")
def eigh(X, U_decision = False, alpha = None, svd = False, n_jobs = 1, overwrite = False):
    """
    Returns sorted eigenvalues and eigenvectors from large to small of
    a symmetric square matrix X. Follows SVD convention. Also flips
    signs of eigenvectors using svd_flip. Uses the Epsilon Jitter 
    Algorithm to guarantee convergence. Allows Ridge Regularization
    default 1e-6.
    [Added 21/11/18] [Edited 24/11/18 Added Complex Support, Eigh alpha
    set to 0 since Eigh errors are rare.]

    Parameters
    -----------
    X:          Symmetric Square Matrix.
    U_decision: Always set to False. Can choose None for no swapping.
    alpha:      Ridge alpha regularization parameter. Default 1e-6.
    svd:        Returns sqrt(W) and V.T
    n_jobs:     Whether to use more >= 1 CPU
    overwrite:  Whether to directly alter the original matrix.

    Returns
    -----------    
    W:          Eigenvalues
    V:          Eigenvectors
    """
    alpha = 1e-6
    n = X.shape[0]
    byte = X.itemsize
    dtype = X.dtype

    evd, evr = eigh_lwork(dtype, byte, n, n)
   
    free = available_memory()
    if evd > free:
        if evr > free:
            raise MemoryError(f"SYEVR requires {evr} MB, but {free} MB is free, "
    f"so an extra {evr-free} MB is required.")
        evd = False
    else:
        evd = True
    
    size = n + 1 # add items to diagonal
    # From Modern Big Data Algorithms: SYEVD mostly faster than SYEVR
    # contradicts MKL's findings
    if evd:
        decomp = lapack("heevd") if isComplex(dtype) else lapack("syevd")
        W, V = do_until_success(decomp, add_jitter, size, overwrite, None, 
            a = X, lower = 0, overwrite_a = overwrite)
    else:
        decomp = lapack("heevr") if isComplex(dtype) else lapack("syevr")
        W, V = do_until_success(decomp, add_jitter, size, overwrite, None, 
            a = X, uplo = "U", overwrite_a = overwrite)

    # return with SVD convention: sort eigenvalues
    svd_flip(None, V, U_decision = U_decision, n_jobs = n_jobs)

    # if svd -> return V.T and sqrt(S)
    if svd:
        for i in range(W.size):
            if W[i] > 0: break
        W[:i] = 0

    if U_decision is not None:
        W, V = W[::-1], V[:,::-1]

    if svd:
        W **= 0.5
        V = transpose(V, True, dtype)
    return W, V


###
@process(memcheck = "extra")
def eig(X, U_decision = False, alpha = None, turbo = True, svd = False, n_jobs = 1, conjugate = True, overwrite = False):
    """
    Returns sorted eigenvalues and eigenvectors from large to small of
    a general matrix X. Follows SVD convention. Also flips signs of 
    eigenvectors using svd_flip. Uses the Epsilon Jitter 
    Algorithm to guarantee convergence. Allows Ridge Regularization
    default 1e-6.

    According to [`Matrix Computations, Third Edition, G. Holub and C. 
    Van Loan, Chapter 5, section 5.4.4, pp 252-253.`], QR is better if
    n >= 5/3p. In Modern Big Data Algorithms, I find QR is better for
    all n > p.
    [Added 21/11/18] [Edited 22/11/18 with turbo -> approximate
    eigendecomposition when p >> n] [Edited 24/11/18 Added Complex Support]

    Parameters
    -----------
    X:          General Matrix.
    U_decision: Always set to False. Can choose None for no swapping.
    alpha:      Ridge alpha regularization parameter. Default 1e-6.
    turbo:      If True, if p >> n, then will output approximate eigenvectors
                where V = (X.T @ U) / sqrt(W)
    svd:        Returns sqrt(W) and V.T
    n_jobs:     Whether to use more >= 1 CPU
    conjugate:  Whether to inplace conjugate but and inplace return original.
    overwrite:  Whether to conjugate transpose inplace.

    Returns
    -----------
    W:          Eigenvalues
    V:          Eigenvectors
    """
    n, p = X.shape
    byte = X.itemsize
    dtype = X.dtype
    a = transpose(X, conjugate, dtype)

    # check memory usage
    free = available_memory()
    evd, evr = eigh_lwork(dtype, byte, n, p)
    eigh_work = _min(evd, evr)

    use_svd = False
    if eigh_work > free:
        use_svd = True
        # check SVD since less memory usage
        # notice since QR used, upper triangular
        MIN = _min(n,p)
        gesdd, gesvd = svd_lwork(dtype, byte, MIN, p)
        gesddT, gesvdT = svd_lwork(dtype, byte, p, MIN) # also check transpose
        svd_work = min(gesdd, gesvd, eigh_work, gesddT, gesvdT)

        if svd_work > free:
            raise MemoryError(f"EIG requires {svd_work} MB, but {free} MB is free, "
    f"so an extra {svd_work-free} MB is required.")

    if not use_svd:
        # From Modern Big Data Algorithms for p >= 1.1n
        if turbo and p >= 1.1*n:
            # Form XXT
            cov = blas("syrk")(a = a, trans = True, alpha = 1)
            W, V = eigh(cov, U_decision = None, overwrite = True)

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

            V = (a @ V) / W**0.5
        else:
            # Form XTX
            cov = blas("syrk")(a = a, trans = False, alpha = 1)
            W, V = eigh(cov, U_decision = None, overwrite = True)

    else:
        __, W, V = svd( qr(X, R_only = True), U_decision = None, overwrite = True)
        if svd:
            return S, V
        W **= 2
        V = transpose(V, True, dtype)

    # revert matrix X back
    if not overwrite and conjugate and isComplex(dtype):
        transpose(X, True, dtype);

    # return with SVD convention: flip signs
    svd_flip(None, V, U_decision = U_decision, n_jobs = n_jobs)

    # if svd -> return V.T and sqrt(S)
    if svd:
        for i in range(W.size):
            if W[i] > 0: break
        W[:i] = 0

        W **= 0.5
        V = transpose(V, True, dtype)

    if U_decision is not None:
        W, V = W[::-1], V[:,::-1]

    return W, V
