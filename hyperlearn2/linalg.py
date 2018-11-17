
from .base import lapack, blas, process
from .utils import do_until_success, triu
import scipy.linalg as scipy
from .numba import jit

###
@process(square = True, memcheck = "columns")
def cholesky(X, alpha = None):
	"""
	[Added 15/11/2018] [Edited 16/11/2018 Numpy is slower. Uses LAPACK only]
	Uses Epsilon Jitter Solver to compute the Cholesky Decomposition
	until success. Default alpha ridge regularization = 1e-8.

	input:		1 argument, 2 optional
	----------------------------------------------------------
	X:			Matrix to be decomposed. Has to be symmetric.
	alpha:		Ridge alpha regularization parameter. Default 1e-6
	turbo:		Boolean to use float32, rather than more accurate float64.

	returns: 	Upper triangular cholesky factor (U)
	----------------------------------------------------------
	"""
	decomp = lapack("potrf", None)
	U = do_until_success(decomp, alpha, X)
	return U

###
@process(square = True, memcheck = "columns")
def cho_solve(X, rhs, alpha = None):
	"""
	[Added 15/11/2018]
	Given U from a cholesky decompostion and a RHS, find a least squares
	solution.

	input:		1 argument, 2 optional
	----------------------------------------------------------
	X:			Matrix to be decomposed. Has to be symmetric.
	alpha:		Ridge alpha regularization parameter. Default 1e-8
	turbo:		Boolean to use float32, rather than more accurate float64.

	returns: 	Upper triangular cholesky factor (U)
	----------------------------------------------------------
	"""
	theta = lapack("potrs")(X, rhs)[0]
	return theta

###
@process(square = True, memcheck = "squared")
def cho_inv(X, turbo = True):
	"""
	[Added 17/11/2018]
	Computes an inverse to the Cholesky Decomposition.

	input:		1 argument, 1 optional
	----------------------------------------------------------
	X:			Upper Triangular Cholesky Factor U. Use cholesky.
	turbo:		Boolean to use float32, rather than more accurate float64.

	returns: 	Upper Triangular Inverse(X)
	----------------------------------------------------------
	"""
	inv = lapack("potri", None, turbo)(X)
	return inv

###
@process(memcheck = "full")
def pinvc(X, alpha = None, turbo = True):
	"""
	[Added 17/11/2018]
	Returns the Pseudoinverse of the matrix X using Cholesky Decomposition.
	Fastest pinv(X) possible, and uses the Epsilon Jitter Algorithm to
	gaurantee convergence. Allows Ridge Regularization - default 1e-8.

	input:		1 argument, 2 optional
	----------------------------------------------------------
	X:			Upper Triangular Cholesky Factor U. Use cholesky.
	alpha:		Ridge alpha regularization parameter. Default 1e-8
	turbo:		Boolean to use float32, rather than more accurate float64.

	returns: 	Pseudoinverse of X. Allows X @ pinv(X) = I, pinv(X) @ X = I.
	----------------------------------------------------------
	"""
	n, p = X.shape
	# determine under / over-determined
	XXT = True if n > p else False
	a = X.T

	# get covariance or gram matrix
	cov = blas("syrk")(a = a, trans = XXT, alpha = 1)

	decomp = lapack("potrf", None)
	U = do_until_success(decomp, alpha, cov)
	U = lapack("potri", None, turbo)(U, overwrite_c = True)[0]

	# if XXT -> XT * (XXT)^-1
	# if XTX -> (XTX)^-1 * XT
	return blas("symm")(a = U, b = a, side = XXT, alpha = 1)


###
@process(memcheck = {"X":"full", "L_only":"same", "U_only":"same"})
def lu(X, L_only = False, U_only = False, overwrite = False):
	"""
	[Added 16/11/2018]
	Computes the pivoted LU decomposition of a matrix. Optional to output
	only L or U components with minimal memory copying.

	input:		1 argument, 3 optional
	----------------------------------------------------------
	X:			Matrix to be decomposed. Can be retangular.
	L_only:		Output only L.
	U_only:		Output only U.
	overwrite:	Whether to directly alter the original matrix.

	returns: 	(L,U) or (L) or (U)
	----------------------------------------------------------
	"""
	n, p = X.shape
	if L_only or U_only:
		A, P, __ = lapack("getrf")(X, overwrite_a = overwrite)
		if L_only:
			###
			@jit  # Output only L part. Overwrites LU matrix to save memory.
			def L_process(n, p, L):
				wide = (p > n)
				k = p

				if wide:
					# wide matrix
					# L get all n rows, but only n columns
					L = L[:, :n]
					k = n

				# tall / wide matrix
				for i in range(k):
					li = L[i]
					li[i+1:] = 0
					li[i] = 1
				# Set diagonal to 1
				return L, k
			A, k = L_process(n, p, A)
			# inc = -1 means reverse order pivoting
			A = lapack("laswp")(a = A, piv = P, inc = -1, k1 = 0, k2 = k-1, overwrite_a = True)
		else:
			# get only upper triangle
			A = triu(n, p, A)
		return A
	else:
		return scipy.lu(X, permute_l = True, check_finite = False, overwrite_a = overwrite)

###
@process(memcheck = {"X":"full", "Q_only":"same", "R_only":"same"})
def qr(X, Q_only = False, R_only = False, overwrite = False):
	"""
	[Added 16/11/2018]
	Computes the reduced economic QR Decomposition of a matrix. Optional
	to output only Q or R.

	input:		1 argument, 3 optional
	----------------------------------------------------------
	X:			Matrix to be decomposed. Can be retangular.
	Q_only:		Output only Q.
	R_only:		Output only R.
	overwrite:	Whether to directly alter the original matrix.

	returns: 	(Q,R) or (Q) or (R)
	----------------------------------------------------------
	"""
	if Q_only or R_only:
		n, p = X.shape
		R, tau, __, __ = lapack("geqrf")(X, overwrite_a = overwrite)

		if Q_only:
			if p > n:
				R = R[:, :n]
			# Compute Q
			Q, __, __ = lapack("orgqr")(R, tau, overwrite_a = True)
			return Q
		else:
			# get only upper triangle
			R = triu(n, p, R)
			return R

	return lapack(None, "qr")(X)
