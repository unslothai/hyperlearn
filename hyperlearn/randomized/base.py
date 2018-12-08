
from .. import linalg
import numpy as np
from ..numba import _min, jit, prange
from ..base import *
from ..utils import *
from ..linalg import transpose
from ..random import normal


########################################################
## 1. ColumnSelect
## 2. Count Sketch (Fast JLT alternative)
## 3. Fast Count Sketch multiply
## 4. ColumnSelect Matrix Multiply (Nystrom Method)

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
@jit
def make_S_sketch(k ,n, sign, position):
    S = np.zeros((k, n), dtype = np.dtype("int8"))
    ones = np.ones(n, dtype = np.dtype("int8"))
    ones[sign] = -1

    for i in range(n):
        S[position[i], i] = ones[i]
    return S

###
def sketch(n, p, k = "auto", output = "sparse"):
    """
    Produces a CountSketch matrix which is similar in nature to
    the Johnson–Lindenstrauss Transform (eps Fast-JLT) as shown
    in Sklearn. But, as shown by David Woodruff, "Sketching as a 
    Tool for Numerical Linear Algebra" [arxiv.org/abs/1411.4357],
    super fast matrix multiplies can be done. Notice a difference
    is HyperLearn's  K = min(n, p)/2, but david says k^2/eps is
    needed (too memory taxing). [Added 4/12/18]

    Parameters
    -----------
    n:              Number of rows
    p:              Number of columns
    k:              (auto, int). Auto is min(n, p) / 2
    output:         (sparse, full). Sparse outputs 2 vectors of
                    position and sign. Full outputs full count
                    sketch matrix.
    Returns
    -----------
    ((position, sign), matrix S) depending on output option.
    """
    if k == "auto":
        k = _min(n, p) / 2
    k = int(k)

    sign = np.random.randint(0, 2, size = n, dtype = bool)
    position = np.random.randint(0, k, size = n)

    if output == "full":
        return make_S_sketch(k, n, sign, position)
    else:
        return position, sign

###
@jit(parallel = True)
def sketch_multiply(S, X):
    """
    Using the fact that S is sparse, SX can be computed in
    approx NNZ(X) time, or O(kp) time.
    """
    k, n = S.shape
    out = np.zeros((k, X.shape[1]), dtype = X.dtype)
    
    for i in prange(k):
        s = S[i]
        for j in prange(n):
            if s[j] != 0:
                out[i] += X[j]*s[j]
    return out

###
@jit(parallel = True)
def sparse_sketch_multiply(k ,position, sign, X):
    """
    Using the fact that S is sparse, SX can be computed in
    approx NNZ(X) time, or O(kp) time.
    """
    n, p = X.shape
    out = np.zeros((k, p), dtype = X.dtype)
    
    for pos in prange(n):
        row = position[pos]
        if sign[pos]:
            out[row] -= X[pos]
        else:
            out[row] += X[pos]
    return out


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
