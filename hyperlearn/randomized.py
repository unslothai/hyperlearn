
from .linalg import transpose
from . import linalg
import numpy as np
from .numba import _min, jit
from .random import normal
from .base import *
from .utils import *

###
@jit
def proportion(X):
    s = 0
    n = X.shape[0]
    for i in range(n):
        s += X[i]
    X /= s
    return X


###
def select(
    X, n_components = 2, solver = "euclidean", output = "columns", 
    duplicates = False, n_oversamples = "klogk", axis = 0):
    """
    Selects columns from the matrix X using many solvers. Also
    called ColumnSelect or LinearSelect, HyperLearn's select allows
    randomized algorithms to select important columns.
    [Added 30/11/18] [Edited 2/12/18 Added BSS sampling]

    Parameters
    -----------
    X:              General Matrix.
    n_components:   How many columns you want.
    solver:         (euclidean, uniform, leverage, BSS) Selects columns
                    based on separate squared norms of each property.
    output:         (columns, statistics, indices) Whether to output actual
                    columns or just indices of the selected columns.
                    Also can choose statistics to get norms only.
    duplicates:     If True, then leaves duplicates as is. If False,
                    uses sqrt(count) as a scaling factor.
    n_oversamples:  (klogk, None, k) How many extra samples is taken.
                    Default = k*log2(k) which guarantees (1+e)||X-X*||
                    error.
    axis:           (0, 1, 2) 0 is columns. 1 is rows. 2 means both.

    Returns
    -----------
    (X*, indices) Depends on output option.
    """
    n, p = X.shape
    dtype = X.dtype

    k = n_components
    if type(n_components) is float and n_components <= 1:
        k = int(k * p) if axis == 0 else int(k * n)
    else:
        k = int(k)
        if axis == 0: 
            if k > p: k = p
        elif axis == 1: 
            if k > n: k = n
    n_components = k

    # If BSS Sampling:
    # Optimal CUR Matrix Decompositions - David Woodruff
    if solver == "BSS": bss = True
    else: bss = False

    # Oversample ratio. klogk allows (1+e)||X-X*|| error.
    if n_oversamples == "klogk" or bss:
        k = int(n_components * np.log2(n_components))
    else:
        k = int(n_components)

    # rows and columns
    if axis == 0:   r, c = False, True
    elif axis == 1: r, c = True, False
    else:           r, c = True, True

    # Calculate row or column importance.
    if solver == "leverage" or bss:
        if axis == 2:
            U, _, VT = svd(X, n_components = n_components)
            row = row_norm(U)
            col = col_norm(VT)
        elif axis == 0:
            _, V = eig(X, n_components = n_components)
            col = row_norm(V)
        else:
            U, _, _ = svd(X, n_components = n_components)
            row = row_norm(U)

    elif solver == "uniform":
        if r:   row = np.ones(n, dtype = dtype)
        if c:   col = np.ones(p, dtype = dtype)
    else:
        if r:   row = row_norm(X)
        if c:   col = col_norm(X)

    if r:       row = proportion(row)
    if c:       col = proportion(col)

    if output == "statistics":
        if axis == 2:       return col, row
        elif axis == 0:     return col
        else:               return row

    # Make a probability drawing distribution.
    if r:
        N = range(n)
        indicesN = np.random.choice(N, size = k, p = row)

        if bss:
            prob = proportion(row[indicesN])
            indicesN = np.random.choice(indicesN, size = n_components, p = prob)
            k = n_components

        # Use extra sqrt(count) scaling factor for repeated columns
        if not duplicates:
            indicesN, countN = np.unique(indicesN, return_counts = True)
            scalerN = countN.astype(dtype) / (row[indicesN] * k)
        else:
            scalerN = 1 / (row[indicesN] * k)
        scalerN **= 0.5
        scalerN = scalerN[:,np.newaxis]
    if c:
        P = range(p)
        indicesP = np.random.choice(P, size = k, p = col)

        if bss:
            prob = proportion(col[indicesP])
            indicesP = np.random.choice(indicesP, size = n_components, p = prob)
            k = n_components

        # Use extra sqrt(count) scaling factor for repeated columns
        if not duplicates:
            indicesP, countP = np.unique(indicesP, return_counts = True)
            scalerP = countP.astype(dtype) / (col[indicesP] * k)
        else:
            scalerP = 1 / (col[indicesP] * k)
        scalerP **= 0.5

    # Output columns if specified
    if output == "columns":
        if c:
            C = X[:,indicesP] * scalerP
        if r:
            R = X[indicesN] * scalerN

        if axis == 2:       return C, R
        elif axis == 0:     return C
        else:               return R
    # Or just output indices with scalers
    else:
        if axis == 2:       return (indicesN, scalerN), (indicesP, scalerP)
        elif axis == 0:     return indicesP, scalerP
        else:               return indicesN, scalerN


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
def randomized_projection(
    X, n_components = 2, solver = 'lu', max_iter = 4, symmetric = False):
    """
    Projects X onto some random eigenvectors, then using a special
    variant of Orthogonal Iteration, finds the closest orthogonal
    representation for X.
    [Added 25/11/18] [Edited 26/11/18 Overwriting all temp variables]
    [Edited 1/12/18 Added Eigh support]

    Parameters
    -----------
    X:              General Matrix.
    n_components:   How many eigenvectors you want.
    solver:         (auto, lu, qr, None) Default is LU Decomposition
    max_iter:       Default is 4 iterations.
    symmetric:      If symmetric, reduces computation time.

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

    # Notice overwriting doesn't matter since guaranteed to converge.
    _solver =                       lambda x: linalg.lu(x, L_only = True, overwrite = True)
    if solver == 'qr': _solver =    lambda x: linalg.qr(x, Q_only = True, overwrite = True)
    elif solver is None or \
        solver == 'None': _solver = lambda x: x / col_norm(x)**0.5

    if symmetric:
        # Get normal random numbers Q~N(0,1)
        # First check if symmetric:
        if X.flags["F_CONTIGUOUS"]:
            X = reflect(X)

        Q = normal(0, 1, (n_components, p), dtype)
        Q /= (row_norm(Q)**0.5)[:,np.newaxis] # Normalize columns
        Q = Q.T

        for __ in range(max_iter):
            Q = linalg.matmul("S @ Y", X, Q)
            Q = _solver(Q)
        Q = linalg.matmul("S @ Y", X, Q)
    else:
        # Get normal random numbers Q~N(0,1)
        Q = normal(0, 1, (p, n_components), dtype)
        Q /= col_norm(Q)**0.5 # Normalize columns

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
@process(memcheck = "truncated")
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
    k_col = k if type(k) is int else int(k*p)
    k_row = k if type(k) is int else int(k*n)
    k = k if type(k) is int else int(k * _min(n, p))

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
        # to compute full spectrum.

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

        # Double pass (reduce number of rows)
        indicesTemp, scalerTemp = select(C, n_components = k_col, axis = 1, output = "indices")
        CC = C[indicesTemp] * scalerTemp

        XTemp = X[indicesTemp] * row[indicesTemp][:,np.newaxis]
        CCX = CC @ linalg.pinvc(CC) @ XTemp
        c = proportion(col_norm(XTemp - CCX))

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















