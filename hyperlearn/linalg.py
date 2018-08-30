from .base import *
from scipy.linalg import pinvh as scipy_pinvh, svd as scipy_svd, pinv2, eigh
from torch import svd

__all__ = ['pinv', 'pinvh', 'svd', 'diagonal', 'cov', 'invCov', 'Vsigma']

@check
def pinv(X):
	"""
	Computes the pseudoinverse of any matrix using SVD if use_gpu
	or if using CPU, using Scipy's pinv2
	"""
	if use_gpu:
		U, S, VT = svd(X)
		cond = S < eps(X)*constant(S[0])
		_S = 1.0 / S
		_S[cond] = 0.0
		VT *= T(_S)
		return dot( T(VT), T(U) )
	return pinv2(X, return_rank = False, check_finite = False)


def pinvh(X):
	"""
	Computes the pseudoinverse of a Hermitian Matrix
	(Positive Symmetric Matrix) using Scipy (fastest)
	"""
	A = X.numpy() if isTensor(X) else X
	return scipy_pinvh(A, check_finite = False, return_rank = False)


@check
def svd(X):
	"""
	Computes the singular value decomposition of a matrix.
	Uses scipy when use_gpu = False, else pytorch is used.
	"""
	if use_gpu: return svd(X)
	return scipy_svd(X, check_finite = False, return_rank = False)


def diagonal(p, multiplier, dtype):
	"""
	Crates a diagonal of ones multiplied by some factor.
	"""
	if use_gpu: return  diag(ones(p) * multiplier).type(dtype)
	return np_diag(np_ones(p) * multiplier).astype(dtype)


def cov(X):
	"""
	Creates the covariance matrix for X.
	Estimated cov(X) = XTX
	"""
	if use_gpu: return X.t().matmul(X)
	return X.T.dot(X)

@check
def invCov(X, alpha = 0, epsilon = True):
	"""
	Calculates the pseudoinverse of the covariance matrix.
	
	invCov also has a epsilon regularization term alpha.
	Originally set to machine precision, this is to counteract
	strange computational errors effectively.

	So, we calculate pinvh(XTX + alphaI) as Hermitian matrix
	(positive symmetric)

	If X is float32, then alpha = 1e-6
	"""
	XTX = cov(X)
	res = resolution(X) if alpha == 0 and epsilon else alpha
	regularizer = diagonal(XTX.shape[0], res, X.dtype)

	if use_gpu: 
		return pinvh(  (XTX + regularizer).numpy() )
	return pinvh( XTX + regularizer  )


@check
def Vsigma(X):
	"""
	Computes efficiently V and S for svd(X) = U * S * VT
	Skips computation of U entirely, by performing
	eigh (eigendecomp) on covariance matrix XTX.

	Then, notice how XTX = V * lambda * VT, thus:
	singular_values S is just lambda**0.5, and V is found.

	Returns S, VT by convention of U * S * VT
	"""
	XTX = cov(X)
	if use_gpu: XTX = XTX.numpy()

	S, V = eigh(XTX, check_finite = False)
	S[S < 0] = 0.0
	S **= 0.5

	S = S[::-1]
	VT = V[:,::-1].T

	return S, VT

