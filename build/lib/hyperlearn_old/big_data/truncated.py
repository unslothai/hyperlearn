
from scipy.sparse import linalg as sparse
from numpy import finfo
from ..utils import *
from ..random import uniform

__all__ = ['truncatedEigh', 'truncatedSVD', 'truncatedEig']


def truncatedEigh(XTX, n_components = 2, tol = None, svd = False, which = 'largest'):
	"""
	[Edited 6/11/2018 Added smallest / largest command]
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
	n_components = int(n_components)
	n,p = XTX.shape
	dtype = XTX.dtype
	assert n == p
	
	if tol is None: tol = finfo(dtype).resolution
	size = n if p >= n else p  # min(n,p)
	v = uniform(-1, 1, size, dtype = dtype)

	if which == 'largest':
		S2, V = sparse.eigsh(XTX, k = n_components, tol = tol, v0 = v)
	else:
		# Uses shift invert mode to get smallest
		S2, V = sparse.eigsh(XTX, k = n_components, tol = tol, v0 = v, sigma = 0)
	V = eig_flip(V)
	
	# Note ARPACK provides SMALLEST to LARGEST S2. Hence, reverse.
	S2, V = S2[::-1], V[:,::-1]

	if svd:
		S2[S2 < 0] = 0.0
		S2 **= 0.5
		return S2, V.T
	return S2, V


def truncatedEig(X, n_components = 2, tol = None, svd = False, which = 'largest'):
	"""
	[Added 6/11/2018]
	Computes truncated eigendecomposition given any matrix X. Directly
	uses TruncatedSVD if memory is not enough, and returns eigen vectors/values.
	Also argument for smallest eigen components are provided.
	"""
	if memoryXTX(X):
		covariance = _XTX(X.T)
		S, VT = truncatedEigh(covariance, n_components, tol, which = which, svd = svd)
	else:
		__, S, VT = truncatedSVD(X, n_components, tol, which = which)
		if svd:
			return S, VT
		S **= 2
		VT = VT.T
	return S, VT



def truncatedSVD(X, n_components = 2, tol = None, transpose = True, U_decision = False, which = 'largest'):
	"""
	[Edited 6/11/2018 Added which command - can get largest or smallest eigen components]
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
	n_components = int(n_components)
	dtype = X.dtype
	n, p = X.shape
	transpose = True if (transpose and p > n) else False
	if transpose: 
		X, U_decision = X.T, not U_decision

	if tol is None: tol = finfo(dtype).resolution
	size = n if p >= n else p  # min(n,p)
	v = uniform(-1, 1, size, dtype = dtype)

	which = 'LM' if which == 'largest' else 'SM'
	U, S, VT = sparse.svds(X, k = n_components, tol = tol, v0 = v, which = which)

	# Note ARPACK provides SMALLEST to LARGEST S. Hence, reverse.
	U, S, VT = U[:, ::-1], S[::-1], VT[::-1]

	U, VT = svd_flip(U, VT, U_decision = U_decision)
	
	if transpose:
		return VT.T, S, U.T
	return U, S, VT



def truncatedPinv(X, n_components = None, alpha = None):
	"""
	[Added 6/11/2018]
	Implements fast truncated pseudoinverse with regularization.
	Can be used as an approximation to the matrix inverse.
	"""
	if alpha != None: assert alpha >= 0
	alpha = 0 if alpha is None else alpha

	if n_components == None:
		# will provide approx sqrt(p) - 1 components.
		# A heuristic, so not guaranteed to work.
		k = int(sqrt(X.shape[1]))-1
		if k <= 0: k = 1
	else:
		k = int(n_components) if n_components > 0 else 1

	X = _float(X)

	U, S, VT = truncatedSVD(X, n_components)
	U, S, VT = _svdCond(U, S, VT, alpha)
	
	return VT.T * S @ U.T
	