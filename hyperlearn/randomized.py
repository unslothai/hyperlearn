
from .linalg import transpose
from . import linalg
import numpy as np
from .numba import _min
from .random import normal, select
from .base import *
from .utils import eig_condition, svd_flip, svd_condition


###
def matmul(
    pattern, X, n_components = 0.5, solver = "euclidean",
    n_oversamples = "klogk", axis = 1):
    """
    Mirrors hyperlearn.linalg's matmul functionality, but extends it by
    using the randomized ColumnSelect algorithm. This can dramatically
    reduce compute time, but still allows good approximate guarantees.
    [Added 1/12/18]

    Parameters
    -----------
    pattern:        Can include: X.H @ X | X @ X.H
    X:              Compulsory left side matrix.
    n_components:   (int, float). Can be a ratio of total number of
                    columns or rows.
    solver:         (euclidean, uniform, leverage) Selects columns
                    based on separate squared norms of each property.
    n_oversamples:  (klogk, None, k) How many extra samples is taken.
                    Default = k*log2(k) which guarantees (1+e)||X-X*||
                    error.
    axis:           (0, 1). Can be 0 (columns) which reduces the total
                    dimensionality of the data or 1 (rows) which just
                    reduces the compute time of forming XTX or XXT.
    Returns
    -----------
    out:            Special sketch matrix output according to pattern.
    indices:        Output the indices used in the sketching matrix.
    scaler:         Output the scaling factor by which the columns are
                    scaled by.
    """
    dtypeX = X.dtype
    n, p = X.shape
    if axis == 2: axis = 1

    # Use ColumnSelect to sketch the matrix:
    indices, scaler = select(
        X, n_components = n_components, axis = axis, n_oversamples = n_oversamples, 
        duplicates = False, solver = solver, output = "indices")

    if axis == 0:
        # Columns
        A = X[:,indices] * scaler
    else:
        A = X[indices] * scaler
    
    return linalg.matmul(pattern, X = A, Y = None), indices, scaler


###
def randomized_projection(X, n_components = 2, solver = 'lu', max_iter = 4):
    """
    Projects X onto some random eigenvectors, then using a special
    variant of Orthogonal Iteration, finds the closest orthogonal
    representation for X.
    [Added 25/11/8] [Edited 26/11/18 Overwriting all temp variables]

    Parameters
    -----------
    X:              General Matrix.
    n_components:   How many eigenvectors you want.
    solver:         (auto, lu, qr, None) Default is LU Decomposition
    max_iter:       Default is 4 iterations.

    Returns
    -----------
    QR Decomposition of orthogonal matrix.
    """
    n, p = X.shape
    dtype = X.dtype
    n_components = int(n_components)
    if max_iter == 'auto':
        # From Modern Big Data Algorithms --> seems like <= 4 is enough.
        max_iter = 6 if n_components < 0.1 * _min(n, p) else 5

    # Check n_components isn't too large
    if n_components > p: n_components = p

    # Get normal random numbers Q~N(0,1)
    Q = normal(0, 1, (p, n_components), dtype)

    # Notice overwriting doesn't matter since guaranteed to converge.
    _solver =                       lambda x: linalg.lu(x, L_only = True, overwrite = True)
    if solver == 'qr': _solver =    lambda x: linalg.qr(x, Q_only = True, overwrite = True)
    elif solver is None or \
        solver == 'None': _solver = lambda x: x / np.linalg.norm(x, ord = 2, axis = 0)

    for __ in range(max_iter):
        Q = X @ Q
        Q = _solver(Q)
        Q = linalg.matmul("X.H @ Y", X, Q)
        Q = _solver(Q)

    Q = X @ Q
    Q = linalg.qr(Q, Q_only = True, overwrite = True)
    return Q


###
@process(memcheck = "truncated")
def svd(
    X, n_components = 2, max_iter = 'auto', solver = 'lu', n_oversamples = 5, 
    U_decision = False, n_jobs = 1, conjugate = True):
    """
    HyperLearn's Fast Randomized SVD is approx 20 - 40 % faster than
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
    U_decision:     Default = False. If True, uses max from U. If None. don't flip.
    n_jobs:         Whether to use more >= 1 CPU
    conjugate:      Whether to inplace conjugate but inplace return original.

    Returns
    -----------    
    U:              Orthogonal Left Eigenvectors
    S:              Descending Singular Values
    VT:             Orthogonal Right Eigenvectors
    """
    n, p = X.shape
    dtype = X.dtype
    ifTranspose = p > n
    if ifTranspose:
        X = transpose(X, conjugate, dtype)

    Q = randomized_projection(X, n_components + n_oversamples, solver, max_iter)

    B = Q.T @ X
    U, S, VT = linalg.svd(B, U_decision = None, overwrite = True)
    del B
    S = S[:n_components]
    U = Q @ U[:, :n_components]

    if ifTranspose:
        U, VT = transpose(VT, True, dtype), transpose(U, True, dtype)
        if conjugate:
            transpose(X, True, dtype);
        U = U[:, :n_components]
    else:
        VT = VT[:n_components, :]

    # flip signs
    svd_flip(U, VT, U_decision = ifTranspose, n_jobs = n_jobs)

    return U, S, VT


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
    del B

    # Flip signs
    svd_flip(None, V, U_decision = False, n_jobs = n_jobs)

    return W, V


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


