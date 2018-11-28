
from .linalg import *
import numpy as np
from .numba import _min
from .random import normal
from .base import *
from .utils import eig_condition, svd_flip

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
        max_iter = 5 if n_components < 0.1 * _min(n, p) else 4

    Q = normal(0, 1, (p, n_components), dtype)

    # Notice overwriting doesn't matter since guaranteed to converge.
    _solver =                       lambda x: lu(x, L_only = True, overwrite = True)
    if solver == 'qr': _solver =    lambda x: qr(x, Q_only = True, overwrite = True)
    elif solver is None or \
        solver == 'None': _solver = lambda x: x / np.linalg.norm(x, ord = 2, axis = 0)

    for __ in range(max_iter):
        Q = _solver(X @ Q)
        Q = _solver(  matmul("X.H @ Y", X, Q)  )


    return qr(X @ Q, Q_only = True, overwrite = True)


###
#@process(memcheck = "truncated")
def randomizedSVD(
    X, n_components = 2, max_iter = 'auto', solver = 'lu', n_oversamples = 10, 
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
                    better eigenvectors.
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

    U, S, VT = svd(Q.T @ X, U_decision = None, overwrite = True)
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

    return U, S[:n_components], VT


###
@process(memcheck = "minimum")
def randomizedEig(
    X, n_components = 2, max_iter = 'auto', solver = 'lu', 
    n_oversamples = 10, conjugate = True, n_jobs = 1):
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
                    better eigenvectors.
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

    W, V = eig(Q.T @ X, U_decision = None, overwrite = True)
    #W, V = W[::-1], V[:,::-1]
    return W, V

    if ifTranspose:
        # Use Modern Big Data Algorithms determination
        # Condition number just in case W[i] <= 0
        W, V = eig_condition(X, W, V)
        return W, V

        if conjugate:
            transpose(X, True, dtype);
    
    # Flip signs
    svd_flip(None, V, U_decision = False, n_jobs = n_jobs)

    return W[:n_components], V[:n_components]
