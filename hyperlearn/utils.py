
from numpy import uint, newaxis, finfo
from .numba import sign, arange
from psutil import virtual_memory
from .exceptions import FutureExceedsMemory

__all__ = ['svd_flip', 'eig_flip', '_svdCond', '_eighCond',
			'memoryXTX', 'memoryCovariance']

_condition = {'f': 1e3, 'd': 1e6}


def svd_flip(U, VT, U_decision = True):
	"""
	Flips the signs of U and VT for SVD in order to force deterministic output.

	Follows Sklearn convention by looking at U's maximum in columns
	as default.
	"""
	if U_decision:
		max_abs_cols = abs(U).argmax(0)
		signs = sign( U[max_abs_cols, arange(U.shape[1])  ]  )
	else:
		# rows of v, columns of u
		max_abs_rows = abs(VT).argmax(1)
		signs = sign( VT[  arange(VT.shape[0]) , max_abs_rows] )

	U *= signs
	VT *= signs[:,newaxis]
	return U, VT



def eig_flip(V):
    """
	Flips the signs of V for Eigendecomposition in order to 
	force deterministic output.

	Follows Sklearn convention by looking at V's maximum in columns
	as default. This essentially mirrors svd_flip(U_decision = False)
	"""
    max_abs_rows = abs(V).argmax(0)
    signs = sign( V[max_abs_rows, arange(V.shape[1]) ] )
    V *= signs
    return V



def _svdCond(U, S, VT, alpha):
	"""
	Condition number from Scipy.
	cond = 1e-3 / 1e-6  * eps * max(S)
	"""
	t = S.dtype.char.lower()
	cond = (S > (_condition[t] * finfo(t).eps * S[0]))
	rank = cond.sum()
	
	S /= (S**2 + alpha)
	return U[:, :rank], S[:rank], VT[:rank]



def _eighCond(S2, V):
	"""
	Condition number from Scipy.
	cond = 1e-3 / 1e-6  * eps * max(S2)

	Note that maximum could be either S2[-1] or S2[0]
	depending on it's absolute magnitude.
	"""
	t = S2.dtype.char.lower()
	absS = abs(S2)
	maximum = absS[0] if absS[0] >= absS[-1] else absS[-1]

	cond = (absS > (_condition[t] * finfo(t).eps * maximum) )
	S2 = S2[cond]
	
	return S2, V[:, cond]



def memoryXTX(X):
	"""
	Computes the memory usage for X.T @ X so that error messages
	can be broadcast without submitting to a memory error.
	"""
	free = virtual_memory().free * 0.95
	byte = 4 if '32' in str(X.dtype) else 8
	memUsage = X.shape[1]**2 * byte

	return memUsage < free



def memoryCovariance(X):
	"""
	Computes the memory usage for X.T @ X or X @ X.T so that error messages
	can be broadcast without submitting to a memory error.
	"""
	n,p = X.shape
	free = virtual_memory().free * 0.95
	byte = 4 if '32' in str(X.dtype) else 8
	size = p if n > p else n
	
	memUsage = size**2 * byte

	if memUsage > free:
		raise FutureExceedsMemory()
	return

