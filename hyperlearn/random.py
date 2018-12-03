
from .numba import jit
import numpy as np
from .base import isComplex

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
    if type(shape) == int:
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
    mean:       Mean of distribution
    std:        Standard Deviation o distribution.
    shape:      Final shape of random matrix
    dtype:      Data type: supports any type

    Returns
    -----------
    Random Vector or Matrix
    """
    std = abs(std)
    left = mean - std*3 # 3 std means 99.7% data
    right = mean + std*3
    return random(left, right, shape, dtype)

