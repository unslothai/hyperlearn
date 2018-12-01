
from .numba import jit
import numpy as np
from .base import isComplex
from .utils import row_norm, col_norm

###
def random(left, right, shape, dtype = np.float32, size = None):
	"""
	Produces pseudo-random numbers between left and right range.
	Notice much more memory efficient than Numpy, as provides
	a DTYPE argument (float32 supported).
	[Added 24/11/2018] [Edited 26/11/18 Inplace operations more
	memory efficient] [Edited 28/11/18 Supports complex entries]

	Parameters
    -----------
    left:		Lower bound
    right:		Upper bound
    shape:		Final shape of random matrix
    dtype:		Data type: supports any type

    Returns
    -----------
    Random Vector or Matrix
	"""
	if size is not None:
		shape = size
	if type(shape) == int:
		# Only 1 long vector --> easily made.
		return uniform_vector(left, right, shape, dtype)
	if len(shape) == 1:
		return uniform_vector(left, right, shape[0], dtype)
	n, p = shape

	l, r = left/2, right/2
	D = np.zeros(1, dtype = dtype)

	part = uniform_vector(l, r, p, D)
	X = np.tile(part, (n, 1))
	add = uniform_vector(l, r, n, D)[:, np.newaxis]
	mult = uniform_vector(0, 1, p, D)
	X *= mult
	X += add

	# If datatype is complex, produce complex entries as well
	# if isComplex(dtype):
	# 	add = add.real
	# 	add = add*1j
	# 	segment = n // 4
	# 	for i in range(4):
	# 		np.random.shuffle(add)
	# 		left = i*segment
	# 		right = (i+1)*segment
	# 		X[left:right] += add[left:right]
	return X


###
@jit
def uniform_vector(l, r, size, dtype):
	zero = np.zeros(size, dtype = dtype.dtype)
	for j in range(size):
		zero[j] = np.random.uniform(l, r)
	return zero


###
def normal(mean, std, shape, dtype = np.float32):
	"""
	Produces pseudo-random normal numbers with mean and std.
	Notice much more memory efficient than Numpy, as provides
	a DTYPE argument (float32 supported).
	[Added 24/11/2018]

	Parameters
    -----------
    mean:		Mean of distribution
    std:		Standard Deviation o distribution.
    shape:		Final shape of random matrix
    dtype:		Data type: supports any type

    Returns
    -----------
    Random Vector or Matrix
	"""
	std = abs(std)
	left = mean - std*3 # 3 std means 99.7% data
	right = mean + std*3
	return random(left, right, shape, dtype)


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
    [Added 30/11/18]

    Parameters
    -----------
    X:              General Matrix.
    n_components:   How many columns you want.
    solver:         (euclidean, uniform, leverage) Selects columns
                    based on separate squared norms of each property.
    output:         (columns, indices) Whether to output actual
                    columns or just indices of the selected columns.
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
    if type(n_components) is float:
        k = int(k * p) if axis == 0 else int(k * n)
    else:
        k = int(k)
        if axis == 0: 
            if k > p: k = p
        elif axis == 1: 
            if k > n: k = n
    n_components = k

    # Oversample ratio. klogk allows (1+e)||X-X*|| error.
    if n_oversamples == "klogk":
        k = int(n_components * np.log2(n_components))
    else:
        k = int(n_components)

    # rows and columns
    if axis == 0:   r, c = False, True
    elif axis == 1: r, c = True, False
    else:           r, c = True, True

    # Calculate row or column importance.
    if solver == "leverage":
        if axis == 2:
            U, _, VT = linalg.svd(X, n_components = n_components)
            row = row_norm(U)
            col = col_norm(VT)
        elif axis == 0:
            _, V = linalg.eig(X, n_components = n_components)
            col = row_norm(V)
        else:
            U, _, _ = linalg.svd(X, n_components = n_components)
            row = row_norm(U)

    elif solver == "uniform":
        if r:   row = np.ones(n, dtype = dtype)
        if c:   col = np.ones(p, dtype = dtype)
    else:
        if r:   row = row_norm(X)
        if c:   col = col_norm(X)

    # Make a probability drawing distribution.
    if r:
        row = proportion(row)
        indicesN = np.random.choice(range(n), size = k, p = row)
        # Use extra sqrt(count) scaling factor for repeated columns
        if not duplicates:
            indicesN, countN = np.unique(indicesN, return_counts = True)
            scalerN = countN.astype(dtype) / (row[indicesN] * k)
        else:
            scalerN = 1 / (row[indicesN] * k)
        scalerN **= 0.5
        scalerN = scalerN[:,np.newaxis]
    if c:
        col = proportion(col)
        indicesP = np.random.choice(range(p), size = k, p = col)
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

