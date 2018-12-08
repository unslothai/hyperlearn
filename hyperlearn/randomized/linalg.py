
from .base import *
import numpy as np
from .. import linalg
from ..numba import _min, jit, _max
from ..base import *
from ..utils import *
from ..stats import corr, corr_sum

########################################################
## 1. Randomized Projection onto orthogonal bases
## 2. SVD, eig, eigh
## 3. CUR decomposition


###
@process(memcheck = "same")
def pinv(
    X, alpha = None, n_components = "auto", max_iter = 'auto', solver = 'lu', 
    n_jobs = 1, conjugate = True):
    """
    Returns the Pseudoinverse of the matrix X using randomizedSVD.
    Extremely fast. If n_components = "auto", will get the top sqrt(p)+1
    singular vectors.
    [Added 30/11/18]

    Parameters
    -----------
    X:              General matrix X.
    alpha :         Ridge alpha regularization parameter. Default 1e-6
    n_components:   Default = auto. Provide less to speed things up.
    max_iter:       Default is 'auto'. Can be int.
    solver:         (auto, lu, qr, None) Default is LU Decomposition
    n_jobs:         Whether to use more >= 1 CPU
    conjugate:      Whether to inplace conjugate but inplace return original.

    Returns
    -----------    
    pinv(X) :       Randomized Pseudoinverse of X. Approximately allows
                    pinv(X) @ X = I. Not exact.
    """
    dtype = X.dtype
    n, p = X.shape

    if n_components == "auto":
        n_components = int(np.sqrt(p))+1
    n_components = n_components if n_components < p else p

    if n_components > p/2:
        print(f"n_components >= {n_components} will be slow. Consider full pinv or pinvc")


    U, S, VT = svd(
                X, U_decision = None, n_components = n_components, max_iter = max_iter,
                solver = solver, n_jobs = n_jobs, conjugate = conjugate)

    U, _S, VT = svd_condition(U, S, VT, alpha)
    return (transpose(VT, True, dtype) * _S)   @ transpose(U, True, dtype)


###
@process(memcheck = "minimum")
def eig(
    X, n_components = 2, max_iter = 'auto', solver = 'lu', 
    n_oversamples = 5, conjugate = True, n_jobs = 1):
    """
    HyperLearn's Fast Randomized Eigendecomposition is approx 20 - 40 % faster than
    Sklearn's implementation depending on n_components and max_iter.
    [Added 27/11/18]

    Parameters
    -----------
    X:              General Matrix.
    n_components:   How many eigenvectors you want.
    max_iter:       Default is 'auto'. Can be int.
    solver:         (auto, lu, qr, None) Default is LU Decomposition
    n_oversamples:  Samples more components than necessary. Used for
                    convergence purposes. More is slower, but allows
                    better eigenvectors. Default = 5
    conjugate:      Whether to inplace conjugate but inplace return original.
    n_jobs:         Whether to use more >= 1 CPU

    Returns
    -----------
    W:              Eigenvalues
    V:              Eigenvectors
    """
    n, p = X.shape
    dtype = X.dtype
    ifTranspose = p > n
    if ifTranspose:
        X = transpose(X, conjugate, dtype)

    Q = randomized_projection(X, n_components + n_oversamples, solver, max_iter)
    B = Q.T @ X


    if ifTranspose:
        # use SVD instead
        V, W, _ = linalg.svd(B, U_decision = None, overwrite = True)
        W = W[:n_components]
        V = Q @ V[:, :n_components]
        if conjugate:
            transpose(X, True, dtype);
        W **= 2
    else:
        W, V = linalg.eig(B, U_decision = None, overwrite = True)
        W = W[:n_components]
        V = V[:,:n_components]
    del B

    # Flip signs
    svd_flip(None, V, U_decision = False, n_jobs = n_jobs)

    return W, V


###
@process(square = True, memcheck = "minimum")
def eigh(
    X, n_components = 2, max_iter = 'auto', solver = 'lu', 
    n_oversamples = 5, conjugate = True, n_jobs = 1):
    """
    HyperLearn's Fast Randomized Hermitian Eigendecomposition uses
    QR Orthogonal Iteration. 
    [Added 27/11/18]

    Parameters
    -----------
    X:              General Matrix.
    n_components:   How many eigenvectors you want.
    max_iter:       Default is 'auto'. Can be int.
    solver:         (auto, lu, qr, None) Default is LU Decomposition
    n_oversamples:  Samples more components than necessary. Used for
                    convergence purposes. More is slower, but allows
                    better eigenvectors. Default = 5
    conjugate:      Whether to inplace conjugate but inplace return original.
    n_jobs:         Whether to use more >= 1 CPU

    Returns
    -----------
    W:              Eigenvalues
    V:              Eigenvectors
    """
    Q = randomized_projection(
        X, n_components + n_oversamples, solver, max_iter, symmetric = True)

    B = linalg.matmul("S @ Y", X, Q).T
    W, V = linalg.eig(B, U_decision = None, overwrite = True)
    W = W[:n_components]**0.5

    V = V[:,:n_components]
    # Flip signs
    svd_flip(None, V, U_decision = False, n_jobs = n_jobs)

    return W, V



###
@process(memcheck = {"X":"truncated","Q_only":"truncated","R_only":"truncated"})
def qr(
    X, Q_only = False, R_only = False, y = None, n_components = 2, 
    solver = "euclidean", n_oversamples = 5):
    """
    Approximate QR Decomposition using the Gram Schmidt Process. Not very
    accurate, but uses Euclidean Norm sampling from the ColumnSelect algo
    to appropriately select the first columns to orthogonalise. You can
    also set solver to "variance", "correlation" or "euclidean".
    [Added 8/12/18]

    Parameters
    -----------
    X:              General Matrix.
    y:              Optional. Used for "correlation" solver.
    n_components:   How many columns to orthogonalise. Default = 2
    solver:         (euclidean, variance, correlation) Which method
                    to choose columns. Default = euclidean.
    n_oversamples:  Samples more components than necessary. Used for
                    convergence purposes. More is slower, but allows
                    better eigenvectors. Default = 5
    Returns
    -----------
    (Q,R) or (Q) or (R) depends on option Q_only or R_only
    """
    if solver == "correlation":
        if y is None:
            C = corr_sum( corr(X, X) )
        else:
            C = abs( corr(X, y) )
        P = C.argsort()
    elif solver == "variance":
        P = X.var(0).argsort()[::-1]
    else:
        P = col_norm(X).argsort()[::-1]

    # Gram Schmidt process
    n, p = X.shape
    k = n_components + n_oversamples
    if k > p: k = p
    Q = gram_schmidt(X, P, n, k)

    if Q_only:
        return Q.T

    # X = QR -> QT X = QT Q R -> QT X = R
    R = Q @ X
    if R_only:
        return R
    return Q.T, R



###
@process(memcheck = "truncated", fractional = False)
def cur(
    X, n_components = 2, solver = "euclidean", n_oversamples = "klogk", success = 0.5):
    """
    Outputs the CUR Decomposition of a general matrix. Similar
    in spirit to SVD, but this time only uses exact columns
    and rows of the matrix. C = columns, U = some projective
    matrix connecting C and R, and R = rows.
    [Added 2/12/18]

    Parameters
    -----------
    X:              General Matrix.
    n_components:   How many "row eigenvectors" you want.
    solver:         (euclidean, leverage, optimal) Selects columns based 
                    on separate squared norms of each property.
                    
                    Error bounds: (eps = 1-success)
                    nystrom:        Nystrom method (Slightly different)
                                    C @ pinv(C intersect R) @ R
                    euclidean:      ||A - A*|| + eps||A||
                    leverage:       (2 + eps)||A - A*||
                    optimal:        (1 + eps)||A - A*||

    n_oversamples:  (klogk, None, k) How many extra samples is taken.
                    Default = k*log2(k) which guarantees (1+e)||X-X*||
                    error.
    success:        Probability of success. Default = 50%. Higher success
                    rates make the algorithm run slower.

    Returns
    -----------
    C:              Column sample
    U:              Connection between columns and rows
    R:              Row sample
    """
    eps = 1 - success
    eps = 1 if eps > 1 else eps
    eps **= 2

    n, p = X.shape
    dtype = X.dtype
    k = n_components
    k_col = k if type(k) is int else _max( int(k*p), 1)
    k_row = k if type(k) is int else _max( int(k*n), 1)
    k = k if type(k) is int else _max( int(k *_min(n, p)), 1)

    if solver == "euclidean":
        # LinearTime CUR 2015 www.cs.utah.edu/~jeffp/teaching/cs7931-S15/cs7931/5-cur.pdf

        c = int(k_col / eps**2)
        r = int(k_row / eps)
        C = select(X, c, n_oversamples = None, axis = 0)
        r, s = select(X, r, n_oversamples = None, axis = 1, output = "indices")
        R = X[r]*s
        phi = C[r]*s

        CTC = linalg.matmul("X.H @ X", C)
        inv = linalg.pinvh(CTC, reflect = False, overwrite = True)
        U = linalg.matmul("S @ Y.H", inv, phi)

    elif solver == "leverage":
        # LeverageScore CUR 2014

        c = int(k*np.log2(k) / eps)
        C, R = select(X, c, n_oversamples = None, axis = 2, solver = "leverage")
        U = linalg.pinvc(C) @ X @ linalg.pinvc(R)

    elif solver == "nystrom":
        # Nystrom Method. Microsoft 2016 "Kernel Nyström Method for Light Transport"

        c = n_components if n_components < p else p
        r = n_components if n_components < n else n

        c = np.random.choice(range(p), size = c, replace = False)
        r = np.random.choice(range(n), size = r, replace = False)
        c.sort(); r.sort();

        C = X[:,c]
        R = X[r]
        U = linalg.pinvc(C[r])

    elif solver == "optimal":
        # Optimal CUR Matrix Decompositions - David Woodruff 2014
        # Slightly changed - uses double sampling since too taxing
        # to compute full spectrum. (2nd sampling done via Count Sketch)
        # random projection

        k1 = int(k_col * np.log2(k_col))
        k2 = int(k_col/eps)
        K = k_col + k2
        P = range(p)

        col, row = select(
            X, n_components = k, solver = "euclidean", output = "statistics", axis = 2,
            duplicates = True)

        # Get Columns from BSS sampling
        indices1 = np.random.choice(P, size = k1, p = col)
        indices1 = np.random.choice(indices1, size = k_col, p = proportion(col[indices1]) )

        indices1, count1 = np.unique(indices1, return_counts = True)
        scaler1 = count1 / (col[indices1] * k1)
        scaler1 **= 0.5

        # Compute error after projecting onto columns
        C = X[:,indices1] * scaler1

        # Double pass (reduce number of rows) uses Count Sketch
        # ie Fast eps JLT Random Projection
        print(n, p , k)
        position, sign = sketch(n, p, k)
        SX = sparse_sketch_multiply(k, position, sign, X)
        SC = sparse_sketch_multiply(k, position, sign, C)

        # Want X - C*inv(C)*X
        c = SX - SC @ linalg.pinvc(SC) @ SX
        c = proportion(col_norm(c))
        return c, C

        # Select extra columns from column residual norms
        indices2 = np.random.choice(P, size = k2, p = c)
        indicesP = np.hstack((indices1, indices2))

        # Determine final scaling factors for C
        indicesP, countP = np.unique(indicesP, return_counts = True)
        scalerP = countP / (col[indicesP] * K)
        scalerP **= 0.5

        return indicesP, scalerP






        # R = X[indicesP]
        # RTR = linalg.pinvc(R) @ R















