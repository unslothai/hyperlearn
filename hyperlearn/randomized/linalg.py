
import numpy as np
from .. import linalg
from ..numba import _min, jit
from ..utils import *
from ..stats import corr, corr_sum
from ..linalg import transpose
from ..random import normal
from .base import *


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
    duplicates = False, n_oversamples = 0, axis = 0, n_jobs = 1):
    """
    Selects columns from the matrix X using many solvers. Also
    called ColumnSelect or LinearSelect, HyperLearn's select allows
    randomized algorithms to select important columns.
    [Added 30/11/18] [Edited 2/12/18 Added BSS sampling]
    [Edited 10/12/18 Fixed BSS Sampling]

    Parameters
    -----------
    X:              General Matrix.
    n_components:   How many columns you want.
    solver:         (euclidean, uniform, leverage, adaptive) Selects columns
                    based on separate squared norms of each property.
                    [NEW Adaptive]. Iteratively adds parts. (Most accuarate)

    output:         (columns, statistics, indices) Whether to output actual
                    columns or just indices of the selected columns.
                    Also can choose statistics to get norms only.
    duplicates:     If True, then leaves duplicates as is. If False,
                    uses sqrt(count) as a scaling factor.
    n_oversamples:  (klogk, 0) How many extra samples is taken.
                    Default = 0. Not much difference.
    axis:           (0, 1, 2) 0 is columns. 1 is rows. 2 means both.
    n_jobs:         Default = 1. Whether to use >= 1 CPU.

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

    # If adaptive
    adaptive = (solver == "adaptive")

    # Oversample ratio. klogk allows (1+e)||X-X*|| error.
    if n_oversamples == "klogk":
        k = int(n_components * np.log2(n_components))
    else:
        k = int(n_components)

    n_components, k = k, n_components

    # rows and columns
    if axis == 0:   r, c = False, True
    elif axis == 1: r, c = True, False
    else:           r, c = True, True

    # Calculate row or column importance.
    if solver == "leverage" or adaptive:
        if axis == 2:
            U, _, VT = svd(X, n_components = n_components, n_oversamples = 1)
            row = row_norm(U)
            col = col_norm(VT)
        elif axis == 0:
            _, V = eig(X, n_components = n_components, n_oversamples = 1)
            col = row_norm(V)
        else:
            U, _, _ = svd(X, n_components = n_components, n_oversamples = 1)
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
    if not adaptive:
        if r:
            N = range(n)
            indicesN = np.random.choice(N, size = k, p = row)

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

            # Use extra sqrt(count) scaling factor for repeated columns
            if not duplicates:
                indicesP, countP = np.unique(indicesP, return_counts = True)
                scalerP = countP.astype(dtype) / (col[indicesP] * k)
            else:
                scalerP = 1 / (col[indicesP] * k)
            scalerP **= 0.5

    # Adaptive solver from Woodruff's 2014 Optimal CUR Decomp paper.
    # Changed and upgraded into incremental solver. (Approx 1+e||A-A*|| error)
    if adaptive and c:
        kk = int(n/2/np.log2(k+1))

        # Use Woodruff's CountSketch matrix to reduce space complexity.
        S = sketch(n, p, kk, method = "left")
        SX = sketch_multiply_left(X, S, kk, n_jobs = n_jobs)

        # Only want top eigenvector
        _, V = eig(SX, n_components = 1, n_oversamples = 1, U_decision = None)

        # Find maximum column norm -> determinstic algorithm
        norm = proportion(row_norm(V))
        select = norm.argmax()
        C = np.zeros((n, k), dtype = X.dtype)
        scalerP = np.zeros(k, dtype = X.dtype)
        indicesP = np.zeros(k, dtype = int)
        indicesP[0] = select

        s = 1/(norm[select]**0.5)
        scalerP[0] = s
        C[:,0] = X[:,select]*s

        seen = 1
        for i in range(k-1):
            I = i+1
            # Produce S*C, which is a smaller matrix
            SC = sketch_multiply_left(C[:,:I], S, kk, n_jobs = n_jobs)

            # Find projection residual
            # Notice left to right sometimes faster -> so check.
            inv = linalg.pinv(SC)
            left_to_right = linalg.dot(SC, inv, SX, message = True)

            if left_to_right:
                # Better to do CC+ by itself.
                CinvC = SC @ inv
                CinvC *= -1
                CinvC.flat[::CinvC.shape[0]+1] += 1
                norm = col_norm(CinvC @ SX)
            else:
                norm = col_norm(SX -  (SC @ (inv @ SX)) )

            # Convert to probabilities
            norm = proportion(norm)
            select = norm.argmax()
            indicesP[I] = select    
            
            # Have to rescale old slcaers, and update new.
            old = seen**0.5
            seen += 1
            new = seen**0.5
            s = old/new
            C[:,:I] *= s
            scalerP[:I] *= s

            # Update C
            s = 1/((norm[select] * seen)**0.5)
            C[:,I] = X[:,select]*s
            scalerP[I] = s

    # Return output
    if output == "columns":
        if not adaptive:
            if c:
                C = X[:,indicesP] * scalerP
            if r:
                R = X[indicesN] * scalerN

        # # Output columns if specified
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
    solver:         (euclidean, uniform, leverage, adaptive) Selects columns
                    based on separate squared norms of each property.
    n_oversamples:  (klogk, 0, k) How many extra samples is taken.
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


########################################################
## 1. Randomized Projection onto orthogonal bases
## 2. SVD, eig, eigh
## 3. CUR decomposition


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
        # # Get normal random numbers Q~N(0,1)
        Q = normal(0, 1, (n_components, p), dtype)
        Q /= (row_norm(Q)**0.5)[:,np.newaxis] # Normalize columns
        Q = Q.T

        max_iter *= 1.5
        max_iter = int(max_iter)
        # Cause symmetric, more stable, but needs more iterations.

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
    X, alpha = None, n_components = "auto", max_iter = 'auto', solver = 'SATAX', 
    n_jobs = 1, conjugate = True, converge = True):
    """
    Returns the Pseudoinverse of the matrix X using randomizedSVD.
    Extremely fast. If n_components = "auto", will get the top sqrt(p)+1
    singular vectors.
    [Added 30/11/18] [Edited 14/12/18 Added Newton Schulz and 2016 Gower's
    Linearly Convergent Randomized Pseudoinverse - R. M. Gower and Peter 
    Richtarik. "Linearly Convergent Randomized Iterative Methods for 
    Computing the Pseudoinverse", arXiv:1612.06255]

    Parameters
    -----------
    X:              General matrix X.
    alpha :         Ridge alpha regularization parameter. Default 1e-6
    n_components:   Default = auto. Provide less to speed things up.
    max_iter:       Default is 'auto'. Can be int.
    solver:         (auto, satax, newton, sketch, lu, qr, None) Default is SATAX.
                    SATAX is from Gower's 2016 paper. (lu, qr and None) use 2011
                    Halko's Randomized Range Finder. Newton is Newton Schulz solver.
                    SKETCH is Newton + sketching.
    converge:       Default = True. If True, uses a newly discovered approach inspired
                    from Newton Schulz (inv -= 2*inv*X*inv)
    n_jobs:         Whether to use more >= 1 CPU
    conjugate:      Whether to inplace conjugate but inplace return original.

    Returns
    -----------    
    pinv(X) :       Randomized Pseudoinverse of X. Approximately allows
                    pinv(X) @ X = I. Not exact.
    """
    dtype = X.dtype
    n, p = X.shape
    solver = solver.lower()

    if n_components == "auto":
        n_components = int(p**0.5 + 1)
    n_components = n_components if n_components < p else p

    # Gower's SATAX solver. Use's sketch and solve paradigm.
    if solver == "satax":
        if max_iter == "auto":
            # No need to do a lot of iterations. But, can provide stability.
            max_iter = 1

        inv = _min(n, p) * X.T / row_norm(X)**2

        # Tried just selecting via leverage scores, uniform etc. Sketch is best.
        S = sketch(p, n, n_components, "right")
        invS = sketch_multiply(inv, S, n_components, method = "right")
        del S

        BT = (X @ invS).T
        C = BT @ X
        CCT = linalg.matmul("X @ X.H", C)
        invCCT = linalg.pinvh(CCT, overwrite = True)

        for i in range(max_iter):
            R = C @ inv - BT
            inv -= linalg.dot(C.T, invCCT, R)


    # Newton Schulz solver. Needs quite a lot of iterations
    elif solver == "newton":
        if max_iter == "auto":
            max_iter = 10
        
        inv = X.T / (2*frobenius_norm(X))

        # Check X+ (2I - XX+)
        left_to_right = linalg.dot(inv, X, inv, message = True)

        for i in range(max_iter):
            if left_to_right:
                diff = linalg.dot(inv, X, inv)
                inv *= 2
                inv -= diff
            else:
                Xinv = X @ inv
                Xinv *= -1
                Xinv.flat[::Xinv.shape[0]+1] += 2
                inv = inv @ Xinv


    # Newton Schulz solver with sketching. Using too many iterations
    # causes errors to actually explode. So, 1 iteration is OK.
    elif solver == "sketch":
        k = int(n ** 0.5 + 1)
        S = sketch(n, p, k, "left")
        SX = sketch_multiply(X, S, k, method = "left")

        inv = X.T / (2*frobenius_norm(X))

        S_first = sketch(p, n, k, "right")
        S_second = sketch(p, p, n_components, "right")
        S_third = sketch(p, n, n_components, "left")

        # X+S * SX
        invS = sketch_multiply(inv, S_first, k, "right")
        invX = invS @ SX

        # (X+S * SX)S
        invXS = sketch_multiply(invX, S_second, n_components, "right")

        # (X+S * SX)S * SX+
        Sinv = sketch_multiply(inv, S_third, n_components, "left")

        inv *= 2
        inv -= invXS @ Sinv


    # Else, use Halko's Randomized Range Finder for SVD. Least accurate.
    else:
        if n_components > p/2:
            print(f"n_components >= {n_components} will be slow. Consider full pinv or pinvc")

        U, S, VT = svd(
                    X, U_decision = None, n_components = n_components, max_iter = max_iter,
                    solver = solver, n_jobs = n_jobs, conjugate = conjugate)

        U, _S, VT = svd_condition(U, S, VT, alpha)
        inv = (transpose(VT, True, dtype) * _S)   @ transpose(U, True, dtype)

    # I discovered that doing this allows better convergence (uses
    # ideas from Newton Schulz). Works on all methods except SATAX
    if solver != "satax" and converge:
        inv -= 2*linalg.dot(inv, X, inv)
    return inv


###
@process(memcheck = "minimum")
def eig(
    X, n_components = 2, max_iter = 'auto', solver = 'lu', 
    n_oversamples = 5, conjugate = True, n_jobs = 1, U_decision = False):
    """
    HyperLearn's Fast Randomized Eigendecomposition is approx 20 - 40 % faster than
    Sklearn's implementation depending on n_components and max_iter.
    [Added 27/11/18] [Edited 12/12/18 Added U_decision]

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
    U_decision:     (False, None)

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
    svd_flip(None, V, U_decision = U_decision, n_jobs = n_jobs)

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
    [Added 8/12/18] [Edited 12/12/18 Uses argpartititon]

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
    n, p = X.shape
    k = n_components + n_oversamples
    if k > p: k = p
    kk = k*-1

    if solver == "correlation":
        if y is None:
            C = corr_sum( corr(X, X) )
        else:
            C = abs( corr(X, y) )
        P = np.argpartition( C, k )[:k]
    elif solver == "variance":
        P = np.argpartition( X.var(0), kk)[kk:]
    else:
        P = np.argpartition( col_norm(X), kk)[kk:]

    # Gram Schmidt process
    Q = gram_schmidt(X, P, n, k)

    if Q_only:
        return Q.T

    # X = QR -> QT X = QT Q R -> QT X = R
    R = Q @ X
    if R_only:
        return R
    return Q.T, R

