
from numpy.random import uniform as _uniform
from numpy import tile, float32


def uniform(left, right, n, p, dtype = float32):
	"""
	[Added 6/11/2018]

	Produces pseudo-random uniform numbers between left and right range.
	Notice much more memory efficient than Numpy, as provides
	a DTYPE argument (float32 supported).
	"""
	l, r = left/2, right/2
	size = (n, 1)
	
	part = _uniform(l,r, size = p).astype(dtype)
	X = tile(part, size)
	add = _uniform(l,r, size = size).astype(dtype)
	mult = _uniform(0,1, size = size).astype(dtype)
	
	return (X + add) * mult

