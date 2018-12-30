
from .numba import njit
import numpy as np
from .base import isComplex, isList

###
@njit
def cov(size = 100, dtype = np.zeros(1, dtype = np.float32)):
	out = np.zeros((size,size), dtype = dtype.dtype)
	diag = np.random.randint(1, size**2, size = size)
	
	for i in range(size):
		out[i, i] = diag[i]

	for i in range(size-1):
		for j in range(i+1,size):
			rand = np.random.uniform(-size**2, size**2)
			out[i, j] = rand
			out[j, i] = rand
	return out


###
def random(left, right, shape = 1, dtype = np.float32, size = None):
    """
    Produces pseudo-random numbers between left and right range.
    Notice much more memory efficient than Numpy, as provides
    a DTYPE argument (float32 supported).
    [Added 24/11/2018] [Edited 26/11/18 Inplace operations more
    memory efficient] [Edited 28/11/18 Supports complex entries]

    Parameters
    -----------
    left:       Lower bound
    right:      Upper bound
    shape:      Final shape of random matrix
    dtype:      Data type: supports any type

    Returns
    -----------
    Random Vector or Matrix
    """
    D = np.zeros(1, dtype = dtype)

    if size is not None:
        shape = size
    if type(shape) is int:
        # Only 1 long vector --> easily made.
        return uniform_vector(left, right, shape, D)
    if len(shape) == 1:
        return uniform_vector(left, right, shape[0], D)
    n, p = shape

    l, r = left/2, right/2

    part = uniform_vector(l, r, p, D)
    X = np.tile(part, (n, 1))
    add = uniform_vector(l, r, n, D)[:, np.newaxis]
    mult = uniform_vector(0, 1, p, D)
    X *= mult
    X += add

    # If datatype is complex, produce complex entries as well
    # if isComplex(dtype):
    #   add = add.real
    #   add = add*1j
    #   segment = n // 4
    #   for i in range(4):
    #       np.random.shuffle(add)
    #       left = i*segment
    #       right = (i+1)*segment
    #       X[left:right] += add[left:right]
    return X


###
@njit
def uniform_vector(l, r, size, dtype):
    zero = np.zeros(size, dtype = dtype.dtype)
    for j in range(size):
        zero[j] = np.random.uniform(l, r)
    return zero

###
def normal(mean, std, shape = 1, dtype = np.float32):
    """
    Produces pseudo-random normal numbers with mean and std.
    Notice much more memory efficient than Numpy, as provides
    a DTYPE argument (float32 supported).
    [Added 24/11/2018]

    Parameters
    -----------
    mean:       Mean of distribution
    std:        Standard Deviation o distribution.
    shape:      Final shape of random matrix
    dtype:      Data type: supports any type

    Returns
    -----------
    Random Vector or Matrix
    """
    std = abs(std)*3
    left = mean - std # 3 std means 99.7% data
    right = mean + std
    return random(left, right, shape, dtype)


###
@njit
def shuffle(x):
    np.random.shuffle(x)

###
@njit
def boolean(n, p):
    out = np.zeros((n,p), dtype = np.bool_)

    cols = np.zeros(p, dtype = np.uint32)
    for i in range(p):
        cols[i] = i

    half = p // 2
    quarter = n // 4

    for i in range(n):
        if i % quarter == 0:
            np.random.shuffle(cols)

        l = np.random.randint(half)
        r = l + half

        if i % 3 == 0:
            for j in range(l, r):
                out[i, cols[j]] = True
        else:
            for j in range(r, l, -1):
                out[i, cols[j]] = True
    return out


###
def randbool(size = 1):
    """
    Produces pseudo-random boolean numbers.
    [Added 22/12/18]

    Parameters
    -----------
    size:       Default = 1. Can be 1D or 2D shape.

    Returns
    -----------
    Random Vector or Matrix
    """
    if isList(size):
        n, p = size
        out = boolean(*size)
    else:
        out = np.random.randint(0, 2, size, dtype = bool)
    return out


###
@njit
def _choice_p(a, p, size = 1, replace = True):
    """
    Deterministic selection of items in an array according
    to some probabilities p.
    [Added 23/12/18]
    """
    n = a.size
    sort = np.argsort(p) # Sort probabilities, making more probable
                        # to appear first.
    if replace:
        counts = np.zeros(n, dtype = np.uint32)

        # Get only top items that add to size
        seen = 0
        for i in range(n-1, 0, -1):
            j = sort[i]
            prob = np.ceil(p[j] * size)
            counts[j] = prob
            seen += prob
            if seen >= size:
                break

        have = 0
        j = n-1
        out = np.zeros(size, dtype = a.dtype)

        while have < size:
            where = sort[j]
            amount = counts[where]
            out[have:have+amount] = a[where]
            have += amount
            j -= 1 
    else:
        length = size if size < n else n
        out = a[sort[:length]]
    return out


###
@njit
def _choice_a(a, size = 1, replace = True):
    return np.random.choice(a, size = size, replace = replace)


###
def choice(a, size = 1, replace = True, p = None):
    """
    Selects elements from an array a pseudo randomnly based on
    possible probabilities p or just uniformally.
    [Added 23/12/18]

    Parameters
    -----------
    a:          Array input
    size:       How many elements to randomnly get
    replace:    Whether to get elements with replacement
    p:          Probability distribution or just uniform

    Returns
    -----------
    Vector of randomnly chosen elements
    """
    if p is not None:
        out = _choice_p(a, p, size, replace)
    else:
        out = _choice_a(a, size, replace)
    return out


###
def randint(low, high = None, size = 1):
    dtype = uinteger(size)
    return np.random.randint(low, high, size = size, dtype = dtype)
