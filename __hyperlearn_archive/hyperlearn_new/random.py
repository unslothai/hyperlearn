
from .numba.types import *
import numpy as np
from .cfuncs import isComplex, isList, uinteger

###
@jit(**nogil)
def cov(size = 100, dtype = np.float32):
	out = np.zeros((size,size), dtype = dtype)
	diag = np.random.randint(1, size**2, size = size)
	
	for i in range(size):
		out[i, i] = diag[i]

	for i in range(size-1):
		for j in range(i+1,size):
			rand = np.random.uniform(-size**2, size**2)
			out[i, j] = rand
			out[j, i] = rand
	return out


######
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
    if size is not None:
        shape = size
    if type(shape) is int:
        # Only 1 long vector --> easily made.
        return uniform_vector(left, right, shape, dtype)
    if len(shape) == 1:
        return uniform_vector(left, right, shape[0], dtype)
    n, p = shape

    l, r = left/2, right/2

    part = uniform_vector(l, r, p, dtype)
    X = np.tile(part, (n, 1))
    add = uniform_vector(l, r, n, dtype)[:, np.newaxis]
    mult = uniform_vector(0, 1, p, dtype)
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


######
@jit(**nogil)
def uniform_vector(l, r, size, dtype):
    zero = np.zeros(size, dtype = dtype)
    for j in range(size):
        zero[j] = np.random.uniform(l, r)
    return zero

######
@jit(**nogil)
def shuffle(x):
    np.random.shuffle(x)

######
@jit(bool_[:,::1](I64, I64), **nogil)
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


######
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


######
@jit(**nogil)
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


######
@jit(**nogil)
def _choice_a(a, size = 1, replace = True):
    return np.random.choice(a, size = size, replace = replace)


######
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


######
def randint(low, high = None, size = 1):
    dtype = uinteger(size)
    return np.random.randint(low, high, size = size, dtype = dtype)


######
from .cfuncs import MAXIMUM, RAND

cycle = MAXIMUM()
Zmultf = 2.0 / cycle
mult1 = 2**0.5
mult2 = 2.3

######
@jit([(A32, I64, I64, U32), (A64, I64, I64, U32)], **nogil)
def normal_1(out, div, diff, x):
    j = 0
    normal = 2.0
    while normal >= 1.0:
        x = (214013 * x + 2531011) & cycle;  unif_1 = x*Zmultf - 1
        x = (214013 * x + 2531011) & cycle;  unif_2 = x*Zmultf - 1
        normal = unif_1*unif_1 + unif_2*unif_2
    temp1 = unif_1*mult1*(1/(normal + 0.02) - 0.1);  temp2 = unif_2*2.3

    for a in range(div):
        out[j] = -temp1*.4;   j += 1;  out[j] = temp2;    j += 1;     # reflection
        out[j] = -temp2*.75;  j += 1;  out[j] = temp1;    j += 1;
        out[j] = -temp1*.75;  j += 1;  out[j] = temp2*.4; j += 1;
        
        normal = 2.0
        while normal >= 1.0:
            x = (214013 * x + 2531011) & cycle;  unif_1 = x*Zmultf - 1
            x = (214013 * x + 2531011) & cycle;  unif_2 = x*Zmultf - 1
            normal = unif_1*unif_1 + unif_2*unif_2
        temp1 = unif_1*mult1*(1/(normal + 0.02) - 0.1);  temp2 = unif_2*2.3
    if diff >= 1:   out[j] = temp1*.2;   j += 1;
    if diff >= 2:   out[j] = -temp2*.2;  j += 1;
    if diff >= 3:   out[j] = temp1*.7;   j += 1;
    if diff >= 4:   out[j] = -temp2*.7;  j += 1;
    if diff >= 5:   out[j] = (temp1+temp2)/2

        
######
@jit(M32_(I64, I64, I64), **gil)
def normal_32_(n, p, seed):
    x = uint32(seed)
    size = n*p
    out = np.zeros(size, dtype = np.float32)

    diff = size % 6
    div = size // 6
    normal_1(out, div, diff, x)
    return out.reshape((n, p))

        
######
@jit(M32_(I64, I64, I64), **parallel)
def normal_parallel_32(n, p, seed):
    x = uint32(seed)
    out = np.zeros((n,p), dtype = np.float32)
    div = p // 6
    diff = p % 6
    
    for i in prange(n):  normal_1(out[i], div, diff, x+i)
    return out


######
@jit(M32_(I64, I64, I64), **gil)
def normal_32(n, p, seed):
    if n*p > 50000:
        return normal_parallel_32(n, p, seed)
    else:
        return normal_32_(n, p, seed)
        
        
######
@jit(M64_(I64, I64, I64), **gil)
def normal_64_(n, p, seed):
    x = uint32(seed)
    size = n*p
    out = np.zeros(size, dtype = np.float64)

    diff = size % 6
    div = size // 6
    normal_1(out, div, diff, x)
    return out.reshape((n, p))

        
######
@jit(M64_(I64, I64, I64), **parallel)
def normal_parallel_64(n, p, seed):
    x = uint32(seed)
    out = np.zeros((n,p), dtype = np.float64)
    div = p // 6
    diff = p % 6
    
    for i in prange(n):  normal_1(out[i], div, diff, x+i)
    return out


######
@jit(M64_(I64, I64, I64), **gil)
def normal_64(n, p, seed):
    if n*p > 50000:
        return normal_parallel_64(n, p, seed)
    else:
        return normal_64_(n, p, seed)


######
def normal(mean = 0, std = 1, size = (100,3), dtype = np.float32, random_state = -1):
    n, p = size
    if dtype == np.float32:
        return normal_32(n, p, RAND(random_state))
    else:
        return normal_64(n, p, RAND(random_state))

