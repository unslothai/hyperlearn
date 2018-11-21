
from numpy.random import uniform as _uniform
from numpy import tile, float32, zeros, newaxis
from numba import njit


def uniform(left, right, n, p = None, dtype = float32):
	"""
	[Added 6/11/2018]

	Produces pseudo-random uniform numbers between left and right range.
	Notice much more memory efficient than Numpy, as provides
	a DTYPE argument (float32 supported).
	"""
	l, r = left/2, right/2
	dtype = zeros(1, dtype = dtype)

	if p == None:
		# Only 1 long vector --> easily made.
		return uniform_vector(left, right, n, dtype)

	part = uniform_vector(l, r, p, dtype)
	X = tile(part, (n, 1))
	add = uniform_vector(l, r, n, dtype)[:, newaxis]
	mult = uniform_vector(0, 1, n, dtype)[:, newaxis]
	return (X + add) * mult


@njit(fastmath = True, nogil = True, cache = True)
def uniform_vector(l, r, size, dtype):
	zero = zeros(size, dtype = dtype.dtype)
	for j in range(size):
		zero[j] = _uniform(l, r)
	return zero

