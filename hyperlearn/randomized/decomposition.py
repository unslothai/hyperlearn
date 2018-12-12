
from .linalg import *
from .. import linalg

###
@process(memcheck = {"X":"minimum","C_only":"min_left","R_only":"min_right"}, fractional = False)
def cur(
    X, C_only = False, R_only = False, n_components = 2, 
    solver = "euclidean", n_oversamples = "klogk", success = 0.5):
    """
    Outputs the CUR Decomposition of a general matrix. Similar
    in spirit to SVD, but this time only uses exact columns
    and rows of the matrix. C = columns, U = some projective
    matrix connecting C and R, and R = rows.
    [Added 2/12/18] [Edited 9/12/18 Added C_only and R_only]

    Parameters
    -----------
    X:              General Matrix.
    C_only:         Only compute C.
    R_only:         Only compute R.
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
    compute_all = (C_only == R_only)
    compute_C = compute_all or C_only
    compute_R = compute_all or R_only

    n, p = X.shape
    dtype = X.dtype
    k = n_components
    k_col = k if type(k) is int else _max( int(k*p), 1)
    k_row = k if type(k) is int else _max( int(k*n), 1)
    k = k if type(k) is int else _max( int(k *_min(n, p)), 1)

    if solver == "euclidean":
        # LinearTime CUR 2015 www.cs.utah.edu/~jeffp/teaching/cs7931-S15/cs7931/5-cur.pdf

        if compute_C:
            c = int(k_col / eps**2)
            C = select(X, c, n_oversamples = None, axis = 0)

        if compute_R:
            r = int(k_row / eps)
            r, s = select(X, r, n_oversamples = None, axis = 1, output = "indices")
            R = X[r]*s

        if compute_all:
            phi = C[r]*s
            CTC = linalg.matmul("X.H @ X", C)
            inv = linalg.pinvh(CTC, reflect = False, overwrite = True)
            U = linalg.matmul("S @ Y.H", inv, phi)

    elif solver == "leverage":
        # LeverageScore CUR 2014
        c = int(k*np.log2(k) / eps)
        if compute_all:
            C, R = select(X, c, n_oversamples = None, axis = 2, solver = "leverage")

        elif compute_C:
            C = select(X, c, n_oversamples = None, axis = 0, solver = "leverage")

        elif compute_R:
            R = select(X, c, n_oversamples = None, axis = 1, solver = "leverage")

        if compute_all:
            U = linalg.pinvc(C) @ X @ linalg.pinvc(R)

    elif solver == "nystrom":
        # Nystrom Method. Microsoft 2016 "Kernel Nyström Method for Light Transport"
        if compute_C:
            c = n_components if n_components < p else p
            c = np.random.choice(range(p), size = c, replace = False)
            C = X[:,c]

        if compute_R:
            r = n_components if n_components < n else n
            r = np.random.choice(range(n), size = r, replace = False)
            R = X[r]

        if compute_all:
            U = linalg.pinv(C[r])

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









