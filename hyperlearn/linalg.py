
from .base import *
from .utils import *
import scipy.linalg as scipy
from .numba import jit, _max, _min
from . import numba


@jit
def dot_left_right(n, a_b, b_c, c):
    # From left X = (AB)C
    AB = a_b*b_c  # First row of AB
    AB *= n       # n * AB rows
    AB_C = b_c*c  # First row of AB_C
    AB_C *= n     # n * AB_C rows
    left = AB + AB_C

    # From right X = A(BC)
    BC = b_c*c    # First row of BC
    BC *= a_b     # a_b * first row BC
    A_BC = a_b*c  # First row of A_BC
    A_BC *= n     # n times
    right = BC + A_BC

    return left, right

###
def dot(A, B, C, message = False):
    """
    Implements fast matrix multiplication of 3 matrices X = ABC
    From left: X = (AB)C. From right: X = A(BC). This function
    calculates which is faster, and outputs the result.
    [Added 10/12/18] [Edited 13/12/18 Added left or right statement]
    [Edited 20/12/18 Uses numba]

    Parameters
    -----------
    A:          First matrix
    B:          Multiplied with 2nd matrix
    C:          Multiplied with 3rd matrix
    message:    Default = False. If True, doesn't output result, but
                outputs TRUE if left to right, else FALSE right to left.
    Returns
    -----------
    (A@B@C or message)
    """
    n, a_b = A.shape    # A and B share sizes. Size of A determines
                        # final number of rows
    b_c = B.shape[1]
    c = C.shape[1]      # final columns

    left, right = dot_left_right(n, a_b, b_c, c)

    if message:
        return left <= right

    if left <= right:
        return A @ B @ C
    return A @ (B @ C)



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
def matmul(pattern, X, Y = None):
    """
    Using BLAS routines GEMM, SYRK, SYMM, multiplies 2 matrices together
    assuming X has some special structure or the output is special. Supports
    symmetric constructions: X.H @ X and X @ X.H; symmetric multiplies:
    S @ Y.H and Y.H @ S where S is a symmetric matrix; general multiplies:
    X @ Y and X.H @ Y.
    [Added 28/11/18 Changed from transpose since it didn't work]
    [Edited 1/12/18 Added S @ Y] [Edited 17/12/18 Added Y @ S]

    Parameters
    -----------
    pattern:    Can include: X.H @ X | X @ X.H | S @ Y.H | 
                Y.H @ S | X @ Y | X.H @ Y | S @ Y | Y @ S
    X:          Compulsory left side matrix.
    Y:          Optional right side matrix.

    Returns
    -----------
    out:        Special matrix output according to pattern.
    """
    pattern = pattern.upper().replace(' ','')
    dtypeX = X.dtype
    XT = X.T
    n = X.shape[0]
    isComplex_dtypeX = isComplex(dtypeX)
    
    if pattern == "X.H@X":
        if isComplex_dtypeX:
            # BLAS SYRK doesn't work
            out = blas("gemm")(a = XT, b = XT, trans_a = 0, trans_b = 2, alpha = 1)
            out = np.conjugate(out, out = out)
        else:
            # Use BLAS SYRK
            out = blas("syrk")(a = XT, trans = 0, alpha = 1)

    elif pattern == "X@X.H":
        if isComplex_dtypeX:
            # BLAS SYRK doesn't work
            out = blas("gemm")(a = XT, b = XT, trans_a = 2, trans_b = 0, alpha = 1)
            out = np.conjugate(out, out = out)
        else:
            # Use BLAS SYRK
            out = blas("syrk")(a = XT, trans = 1, alpha = 1)

    elif pattern == "X.H@Y":
        if isComplex_dtypeX:
            dtypeY = Y.dtype
            if isComplex_dtypeY:
                out = blas("gemm")(a = XT, b = Y.T, trans_a = 0, trans_b = 2, alpha = 1)
            else:
                out = XT @ Y
            out = np.conjugate(out, out = out)
        else:
            out = XT @ Y

    elif pattern == "X@Y":
        out = X @ Y
        
    # Symmetric Multiply
    # If it's F Contiguous, I assume it's UPPER. If not, it's transposed.
    elif pattern == "S@Y.H":
        dtypeY = Y.dtype
        isComplex_dtypeY = isComplex(dtypeY)

        if isComplex_dtypeY:
            a = X if X.flags["F_CONTIGUOUS"] else XT
            # Symmetric doesn't work
            out = blas("gemm")(a = a, b = Y, trans_b = 2, alpha = 1)
        else:
            YT = Y.T
            if X.flags["F_CONTIGUOUS"]:
                out = blas("symm")(a = X, b = YT, side = 0, alpha = 1)
            else:
                out = blas("symm")(a = XT, b = YT, side = 0, alpha = 1, lower = 1)

    elif pattern == "Y.H@S":
        dtypeY = Y.dtype

        if isComplex_dtypeY:
            a = X if X.flags["F_CONTIGUOUS"] else XT
            # Symmetric doesn't work
            out = blas("gemm")(a = Y, b = a, trans_a = 2, alpha = 1)
        else:
            YT = Y.T
            if X.flags["F_CONTIGUOUS"]:
                out = blas("symm")(a = X, b = YT, side = 1, alpha = 1)
            else:
                out = blas("symm")(a = XT, b = YT, side = 1, alpha = 1, lower = 1)

    elif pattern == "Y@S":
        if X.flags["F_CONTIGUOUS"]:
            out = blas("symm")(a = X, b = Y, side = 1, alpha = 1)
        else:
            out = blas("symm")(a = XT, b = Y, side = 1, alpha = 1, lower = 1)

    elif pattern == "S@Y":
        if X.flags["F_CONTIGUOUS"]:
            out = blas("symm")(a = X, b = Y, side = 0, alpha = 1)
        else:
            out = blas("symm")(a = XT, b = Y, side = 0, alpha = 1, lower = 1)

    else:
        raise NameError(f"Pattern = {pattern} is not recognised.")
    return out


###
@process(square = True, memcheck = "columns")
def cholesky(X, alpha = None, overwrite = False):
    """
    Uses Epsilon Jitter Solver to compute the Cholesky Decomposition
    until success. Default alpha ridge regularization = 1e-6.
    [Added 15/11/18] [Edited 16/11/18 Numpy is slower. Uses LAPACK only]
    [Edited 18/11/18 Uses universal "do_until_success"]

    Parameters
    -----------
    X :         Matrix to be decomposed. Has to be symmetric.
    alpha :     Ridge alpha regularization parameter. Default 1e-6
    overwrite:  Whether to inplace change data.

    Returns
    -----------
    U :         Upper triangular cholesky factor (U)
    """
    decomp = lapack("potrf")
    U = do_until_success(decomp, add_jitter, X.shape[0], overwrite, alpha, X)
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
    inv = lapack("potri", turbo)(X)
    return inv[0]


###
@process(memcheck = "full")
def pinvc(X, alpha = None, turbo = True, overwrite = False):
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
    overwrite:  Whether to overwrite intermmediate results. Will cause
                alpha to be increased by a factor of 10.

    Returns
    -----------    
    pinv(X) :   Pseudoinverse of X. Allows pinv(X) @ X = I if n >= p or X
                @ pinv(X) = I for p > n.
    """
    n, p = X.shape
    # determine under / over-determined
    XTX = n > p
    dtype = X.dtype

    # get covariance or gram matrix
    U = matmul("X.H @ X", X) if XTX else matmul("X @ X.H", X)

    decomp = lapack("potrf")
    U = do_until_success(decomp, add_jitter, _min(n,p), overwrite, alpha, U)
    U = lapack("potri", turbo)(U, overwrite_c = True)[0]

    # if XXT -> XT * (XXT)^-1
    # if XTX -> (XTX)^-1 * XT
    inv = matmul("S @ Y.H", U, X) if XTX else matmul("Y.H @ S", U, X)
    return inv


###
_reflect = reflect
@process(square = True, memcheck = "full")
def pinvch(X, alpha = None, turbo = True, overwrite = False, reflect = True):
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
    overwrite:  Whether to overwrite X inplace with pinvh.
    reflect:    Output full matrix or 1/2 triangular

    Returns
    -----------    
    pinv(X) :   Pseudoinverse of X. Allows pinv(X) @ X = I.
    """
    decomp = lapack("potrf")
    U = do_until_success(decomp, add_jitter, X.shape[0], overwrite, alpha, X)
    U = lapack("potri", turbo)(U, overwrite_c = True)[0]

    return _reflect(U) if reflect else U


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
        A, P, _ = lapack("getrf")(X, overwrite_a = overwrite)
        if L_only:
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
    Computes the pseudoinverse of a square matrix X using LU Decomposition.
    Notice, it's much faster to use pinvc (Choleksy Inverse).
    [Added 18/11/18] [Edited 26/11/18 Fixed ridge regularization]

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
    A, P, _ = lapack("getrf")(X, overwrite_a = overwrite)

    inv = lapack("getri")
    A = do_until_success(inv, U_process, _min(n,p), True, alpha, lu = A, piv = P, overwrite_lu = True) 
    # overwrite shouldnt matter in first go
    return A


###
@process(memcheck = {"X":"full", "Q_only":"same", "R_only":"same"})
def qr(X, Q_only = False, R_only = False, overwrite = False):
    """
    Computes the reduced economic QR Decomposition of a matrix. Optional
    to output only Q or R.
    [Added 16/11/18] [Edited 28/11/18 Complex support]

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
    dtype = X.dtype

    if Q_only or R_only:
        n, p = X.shape
        R, tau, _, _ = lapack("geqrf")(X, overwrite_a = overwrite)

        if Q_only:
            if p > n:
                R = R[:, :n]
            # Compute Q
            if isComplex(dtype):
                Q, _, _ = lapack("ungqr")(R, tau, overwrite_a = True)
            else:
                Q, _, _ = lapack("orgqr")(R, tau, overwrite_a = True)
            return Q
        else:
            # get only upper triangle
            R = triu(n, p, R)
            return R

    return lapack(None, "qr")(X)


###
@jit
def svd_lwork(isComplex_dtype, byte, n, p):
    """
    Computes the work required for SVD (gesdd, gesvd)
    [Updated 20/12/18 Uses Numba for some microsecond saving]
    """
    if n < p:
        MIN = n
        MAX = p
    else:
        MIN = p
        MAX = n

    # check memory usage for GESDD vs GESVD. Use one which is within memory limits.
    if isComplex_dtype:
        gesdd = (MIN + 3)*MIN  # updated from netlib
        gesvd = 2*MIN + MAX
    else:
        # min(n, p)*(6 + min(n,p)) + max(n, p)
        gesdd = (4*MIN + 7)*MIN   # updated from netlib
        a = 3*MIN + MAX
        b = 5*MIN
        gesvd = a if a > b else b

    gesdd *= byte; gesvd *= byte;
    gesdd = int(gesdd); gesvd = int(gesvd)
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
    conjugate:  Whether to inplace conjugate but inplace return original.
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
    isComplex_dtype = isComplex(dtype)

    if ifTranspose: 
        X = transpose(X, conjugate, dtype)
        U_decision = not U_decision
        n, p = X.shape
    byte = X.itemsize

    gesdd, gesvd = svd_lwork(isComplex_dtype, byte, n, p)
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
            U, S, VT, _ = lapack("gesdd")(X, full_matrices = False, overwrite_a = overwrite)
        else:
            U, S, VT = numba.svd(X, full_matrices = False)
    else:
        U, S, VT, _ = lapack("gesvd")(X, full_matrices = False, overwrite_a  = overwrite)
        
    # Return original X if X.H
    if not overwrite and conjugate and isComplex_dtype:
        transpose(X, True, dtype);
    
    # Flip if svd(X.T) was performed.
    if ifTranspose:
        U, VT = transpose(VT, True, dtype), transpose(U, True, dtype)

    # In place flips sign according to max(abs(VT))
    svd_flip(U, VT, U_decision = U_decision, n_jobs = n_jobs)

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
    U, _S, VT = svd_condition(U, S, VT, alpha)
    return (transpose(VT, True, dtype) * _S) @ transpose(U, True, dtype)


###
@jit
def eigh_lwork(isComplex_dtype, byte, n, p):
    """
    Computes the work required for EIGH (syevr, syevd, heevr, heevd)
    SYEVD = 1 + 6n + 2n^2
    SYEVR = 26n
    HEEVD = 2n + n^2
    HEEVR = 2n
    [Updated 20/12/18 Uses Numba for some microsecond saving]
    """
    if p >= 1.1*n:
        s = n
    else:
        s = p

    if isComplex_dtype:
        heevd = 2*s + s**2
        heevr = 2*s + 1
        evd, evr = heevd, heevr
    else:
        syevd = 1 + 6*s + 2*s**2
        syevr = 26*s
        evd, evr = syevd, syevr

    evd *= byte; evr *= byte;
    evd = int(evd); evr = int(evr)
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
    n = X.shape[0]
    byte = X.itemsize
    dtype = X.dtype
    isComplex_dtype = isComplex(dtype)

    evd, evr = eigh_lwork(isComplex_dtype, byte, n, n)
   
    free = available_memory()
    if evd > free:
        if evr > free:
            raise MemoryError(f"SYEVR requires {evr} MB, but {free} MB is free, "
    f"so an extra {evr-free} MB is required.")
        evd = False
    else:
        evd = True
    
    # From Modern Big Data Algorithms: SYEVD mostly faster than SYEVR
    # contradicts MKL's findings
    if evd:
        decomp = lapack("heevd") if isComplex_dtype else lapack("syevd")
        W, V = do_until_success(
            decomp, add_jitter, n, overwrite, None, 
            a = X, lower = 0, overwrite_a = overwrite)
    else:
        decomp = lapack("heevr") if isComplex_dtype else lapack("syevr")
        W, V = do_until_success(
            decomp, add_jitter, n, overwrite, None, 
            a = X, uplo = "U", overwrite_a = overwrite)

    # if svd -> return V.T and sqrt(S)
    if svd:
        W = eig_search(W, 0)

    if U_decision is not None:
        W, V = W[::-1], V[:,::-1]

    if svd:
        W **= 0.5
        V = transpose(V, True, dtype)

    # return with SVD convention: sort eigenvalues
    svd_flip(None, V, U_decision = U_decision, n_jobs = n_jobs)

    return W, V


###
_svd = svd
@process(memcheck = "extra")
def eig(
    X, U_decision = False, alpha = None, turbo = True, svd = False, 
    n_jobs = 1, conjugate = True, overwrite = False, use_svd = False):
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
    conjugate:  Whether to inplace conjugate but inplace return original.
    overwrite:  Whether to conjugate transpose inplace.
    use_svd:    Use SVD instead of EIGH (slower, but more robust)

    Returns
    -----------
    W:          Eigenvalues
    V:          Eigenvectors
    """
    n, p = X.shape
    byte = X.itemsize
    dtype = X.dtype
    isComplex_dtype = isComplex(dtype)

    # check memory usage
    free = available_memory()
    evd, evr = eigh_lwork(isComplex_dtype, byte, n, p)
    eigh_work = _min(evd, evr)

    if eigh_work > free:
        use_svd = True
        # check SVD since less memory usage
        # notice since QR used, upper triangular
        MIN = _min(n,p)
        gesdd, gesvd = svd_lwork(isComplex_dtype, byte, MIN, p)
        gesddT, gesvdT = svd_lwork(isComplex_dtype, byte, p, MIN) # also check transpose
        svd_work = min(gesdd, gesvd, eigh_work, gesddT, gesvdT)

        if svd_work > free:
            raise MemoryError(f"EIG requires {svd_work} MB, but {free} MB is free, "
    f"so an extra {svd_work-free} MB is required.")

    if not use_svd:
        # From Modern Big Data Algorithms for p >= 1.1n
        if turbo and p >= 1.1*n:
            # Form XXT
            cov = matmul("X @ X.H", X)
            W, V = eigh(cov, U_decision = None, overwrite = True) # overwrite doesn't matter

            W, V = eig_condition(X, W, V)
        else:
        # Form XTX
            cov = matmul("X.H @ X", X)
            W, V = eigh(cov, U_decision = None, overwrite = True)
        W, V = W[::-1], V[:,::-1]
        
    else:
        _, W, V = _svd( qr(X, R_only = True), U_decision = None, overwrite = True)
        if svd:
            return W, V
        W **= 2
        V = transpose(V, True, dtype)

    # revert matrix X back
    if not overwrite and conjugate and isComplex(dtype):
        transpose(X, True, dtype);

    # if svd -> return V.T and sqrt(S)
    if svd:
        W = eig_search(W, 0)

        W **= 0.5
        V = transpose(V, True, dtype)

    # return with SVD convention: flip signs
    svd_flip(None, V, U_decision = U_decision, n_jobs = n_jobs)

    return W, V


###
@process(square = True, memcheck = "full")
def pinvh(X, alpha = None, turbo = True, overwrite = False, reflect = True):
    """
    Returns the inverse of a square Hermitian Matrix using Eigendecomposition. 
    Uses the Epsilon Jitter Algorithm to guarantee convergence. 
    Allows Ridge Regularization - default 1e-6.
    [Added 19/11/18]

    Parameters
    -----------
    X :         Upper Symmetric Matrix X
    alpha :     Ridge alpha regularization parameter. Default 1e-6
    turbo :     Boolean to use float32, rather than more accurate float64.
    overwrite:  Whether to overwrite X inplace with pinvh.
    reflect:    Output full matrix or 1/2 triangular

    Returns
    -----------    
    pinv(X) :   Pseudoinverse of X. Allows pinv(X) @ X = I.
    """
    W, V = eigh(X, U_decision = None, alpha = alpha, overwrite = overwrite)

    dtype = V.dtype.char.lower()
    eps = np.finfo(dtype).eps

    if dtype == 'f':
        eps *= 1000
    else:
        eps *= 1000000

    above_cutoff = eigh_search(W, eps)
    _W = 1.0 / W[above_cutoff]
    V = V[:, above_cutoff]

    inv = V * _W @ transpose(V)
    return inv

