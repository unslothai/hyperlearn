
import numpy as np
from .numba import jit, prange
dtypes = (np.uint8, np.uint16, np.uint32, np.uint64)

def uint(i):
	"""
	[Added 14/11/2018]
	Outputs a small array with the correct dtype that minimises
	memory usage for storing the maximal number of elements of
	size i

	input:		1 argument
	----------------------------------------------------------
	i:			Size of maximal elements you want to store

	returns: 	array of size 1 with correct dtype
	----------------------------------------------------------
	"""
	for x in dtypes:
		if np.iinfo(x).max >= i:
			break
	return np.empty(1, dtype = x)


def do_until_success(function, alpha = None, *args, **kwargs):
	X = args[0]
	n = X.shape[0] + 1
	alpha = 1e-6 if alpha == None else alpha
	old_alpha = 0
	error = 1
	while error != 0:
		X.flat[::n] += (alpha - old_alpha)
		try:
			out = function(*args, **kwargs)
			if type(out) == tuple:
				error = out[-1]
				out = out[:-1]
				if len(out) == 1:
					out = out[0]
			else:
				error = 0
		except: 
			pass
		if error != 0:
			old_alpha = alpha
			alpha *= 10

	X.flat[::n] -= alpha
	return out


###
@jit  # Output only triu part. Overwrites matrix to save memory.
def triu(n, p, U):
	tall = (n > p)
	k = n

	if tall:
		# tall matrix
		# U get all p rows
		U = U[:p]
		k = p

	for i in range(1, k):
		U[i, :i] = 0
	return U

###
@jit(parallel = True)
def _reflect(X):
	n = len(X)
	for i in prange(1, n):
		Xi = X[i]
		for j in range(i):
			X[j, i] = Xi[j]
	return X
