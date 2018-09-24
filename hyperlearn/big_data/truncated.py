
from scipy.sparse import linalg as sparse
from numpy import random as _random, finfo
from ..utils import *

__all__ = ['truncatedEigh', 'truncatedSVD']


def truncatedEigh(XTX, n_components = 2, tol = None, svd = False):
	"""
	Computes the Truncated Eigendecomposition of a Hermitian Matrix
	(positive definite). K = 2 for default.
	Return format is LARGEST eigenvalue first.

	If SVD is True, then outputs S2**0.5 and sets negative S2 to 0
	and outputs VT and not V.
	
	Speed
	--------------
	Uses ARPACK from Scipy to compute the truncated decomp. Note that
	to make it slightly more stable and faster, follows Sklearn's
	random intialization from -1 -> 1.

	Also note tolerance is resolution(X), and NOT eps(X)

	Might switch to SPECTRA in the future.
	
	Stability
	--------------
	EIGH FLIP is called to flip the eigenvector signs for deterministic
	output.
	"""
	n,p = XTX.shape
	dtype = XTX.dtype
	assert n == p
	
	if tol is None: tol = finfo(dtype).resolution
	size = n if p >= n else p  # min(n,p)
	v = _random.uniform(-1, 1, size = size ).astype(dtype)

	S2, V = sparse.eigsh(XTX, k = n_components, tol = tol, v0 = v)
	V = eig_flip(V)
	
	# Note ARPACK provides SMALLEST to LARGEST S2. Hence, reverse.
	S2, V = S2[::-1], V[:,::-1]

	if svd:
		S2[S2 < 0] = 0.0
		S2 **= 0.5
		return S2, V.T
	return S2, V



def truncatedSVD(X, n_components = 2, tol = None, transpose = True, U_decision = False):
	"""
	Computes the Truncated SVD of any matrix. K = 2 for default.
	Return format is LARGEST singular first first.
	
	Speed
	--------------
	Uses ARPACK from Scipy to compute the truncated decomp. Note that
	to make it slightly more stable and faster, follows Sklearn's
	random intialization from -1 -> 1.

	Also note tolerance is resolution(X), and NOT eps(X). Also note
	TRANSPOSE is True. This means instead of computing svd(X) if p > n,
	then computing svd(X.T) is faster, but you must output VT.T, S, U.T

	Might switch to SPECTRA in the future.
	
	Stability
	--------------
	SVD FLIP is called to flip the VT signs for deterministic
	output. Note uses VT based decision and not U based decision.
	U_decision can be changed to TRUE for Sklearn convention
	"""
	dtype = X.dtype
	n, p = X.shape
	transpose = True if (transpose and p > n) else False
	if transpose: 
		X, U_decision = X.T, ~U_decision

	if tol is None: tol = finfo(dtype).resolution
	size = n if p >= n else p  # min(n,p)
	v = _random.uniform(-1, 1, size = size ).astype(dtype)

	U, S, VT = sparse.svds(X, k = n_components, tol = tol, v0 = v)

	# Note ARPACK provides SMALLEST to LARGEST S. Hence, reverse.
	U, S, VT = U[:, ::-1], S[::-1], VT[::-1]

	U, VT = svd_flip(U, VT, U_decision = U_decision)
	
	if transpose:
		return VT.T, S, U.T
	return U, S, VT


