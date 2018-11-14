
from scipy.linalg import lu as _lu, qr as _qr
from . import numba
from numba import njit
from .base import *
from .utils import *
from numpy import float32, float64


__all__ = ['cholesky', 'invCholesky', 'pinvCholesky', 'cholSolve',
			'svd', 'lu', 'qr', 
			'pinv', 'pinvh', 
			'eigh', 'pinvEig', 'eig']


def cholesky(XTX, alpha = None, fast = True):
	"""
	Computes the Cholesky Decompsition of a Hermitian Matrix
	(Positive Symmetric Matrix) giving a Upper Triangular Matrix.
	
	Cholesky Decomposition is used as the default solver in HyperLearn,
	as it is super fast and allows regularization. HyperLearn's
	implementation also handles rank deficient and ill-conditioned
	matrices perfectly with the help of the limiting behaivour of
	adding forced epsilon regularization.
	
	Speed
	--------------
	If USE_GPU:
		Uses PyTorch's Cholesky. Speed is OK.
	If CPU:
		Uses Numpy's Fortran C based Cholesky.
		If NUMBA is not installed, uses very fast LAPACK functions.
	
	Stability
	--------------
	Alpha is added for regularization purposes. This prevents system
	rounding errors and promises better convergence rates.
	"""
	n,p = XTX.shape;  assert n==p
	error = 1
	alpha = ALPHA_DEFAULT if alpha == None else alpha
	old_alpha = 0

	decomp = lapack("potrf", fast, "cholesky")

	while error != 0:
		if PRINT_ALL_WARNINGS: 
			print('cholesky Alpha = {}'.format(alpha))

		# Add epsilon jitter to diagonal. Note adding
		# np.eye(p)*alpha is slower and uses p^2 memory
		# whilst flattening uses only p memory.
		addDiagonal(XTX, alpha-old_alpha)
		try:
			cho = decomp(XTX)
			if USE_NUMBA: 
				cho = cho.T
				error = 0
			else:
				cho, error = cho
		except: pass
		if error != 0:
			old_alpha = alpha
			alpha *= 10

	addDiagonal(XTX, -alpha)
	return cho


def cholSolve(A, b, alpha = None):
	"""
	[Added 20/10/2018]
	Faster than direct inverse solve. Finds coefficients in linear regression
	allowing A @ theta = b.
	Notice auto adds epsilon jitter if solver fails.
	"""
	n,p = A.shape;	assert n == p and b.shape[0] == n
	error = 1
	alpha = ALPHA_DEFAULT if alpha is None else alpha
	old_alpha = 0

	solver = lapack("potrs")

	while error != 0:
		if PRINT_ALL_WARNINGS: 
			print('cholSolve Alpha = {}'.format(alpha))

		# Add epsilon jitter to diagonal. Note adding
		# np.eye(p)*alpha is slower and uses p^2 memory
		# whilst flattening uses only p memory.
		addDiagonal(A, alpha-old_alpha)
		try:
			coef, error = solver(A, b)
		except: pass
		if error != 0:
			old_alpha = alpha
			alpha *= 10

	addDiagonal(A, -alpha)
	return coef


def lu(X, L_only = False, U_only = False):
	"""
	[Edited 8/11/2018 Changed to LAPACK LU if L/U only wanted]
	Computes the LU Decomposition of any matrix with pivoting.
	Provides L only or U only if specified.

	Much faster than Scipy if only U/L wanted, and more memory efficient,
	since data is altered inplace.
	"""
	n, p = X.shape
	if L_only or U_only:

		A, P, __ = lapack("getrf")(X)
		if L_only:
			A, k = L_process(n, p, A)
			# inc = -1 means reverse order pivoting
			A = lapack("laswp")(a = A, piv = P, inc = -1, k1 = 0, k2 = k-1, overwrite_a = True)
		else:
			A = triu_process(n, p, A)
		return A
	else:
		return _lu(X, permute_l = True, check_finite = False)


@njit(fastmath = True, nogil = True, cache = True)
def L_process(n, p, L):
	"""
	Auxiliary function to modify data in place to only get L from LU decomposition.
	"""
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


@njit(fastmath = True, nogil = True, cache = True)
def triu_process(n, p, U):
	"""
	Auxiliary function to modify data in place to only get upper triangular
	part of matrix. Used in QR (get R) and LU (get U) decompositon.
	"""
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



def qr(X, Q_only = False, R_only = False, overwrite = False):
	"""
	[Edited 8/11/2018 Added Q, R only parameters. Faster than Numba]
	[Edited 9/11/2018 Made R only more memory efficient (no data copying)]
	Computes the reduced QR Decomposition of any matrix.
	Uses optimized NUMBA QR if avaliable else use's Scipy's
	version.

	Provides Q or R only if specified, and is must faster + more memory
	efficient since data is changed inplace.
	"""
	if Q_only or R_only:
		# Compute R
		n, p = X.shape
		R, tau, __, __ = lapack("geqrf")(X, overwrite_a = overwrite)

		if Q_only:
			if p > n:
				R = R[:, :n]
			# Compute Q
			Q, __, __ = lapack("orgqr")(R, tau, overwrite_a = True)
			return Q
		else:
			# Do nothing, as R is already computed.
			R = triu_process(n, p, R)
			return R

	if USE_NUMBA: return numba.qr(X)
	return _qr(X, mode = 'economic', check_finite = False, overwrite_a = overwrite)



def invCholesky(X, fast = False):
	"""
	Computes the Inverse of a Hermitian Matrix
	(Positive Symmetric Matrix) after provided with Cholesky's
	Lower Triangular Matrix.
	
	This is used in conjunction in solveCholesky, where the
	inverse of the covariance matrix is needed.
	
	Speed
	--------------
	If USE_GPU:
		Uses PyTorch's Triangular Solve given identity matrix. Speed is OK.
	If CPU:
		Uses very fast LAPACK algorithms for triangular system inverses.
	
	Stability
	--------------
	Note that LAPACK's single precision (float32) solver (strtri) is much more
	unstable than double (float64). So, default strtri is OFF.
	However, speeds are reduced by 50%.
	"""
	assert X.shape[0] == X.shape[1]
	choInv = lapack("trtri", fast)(X)[0]

	return choInv @ choInv.T

	

def pinvCholesky(X, alpha = None, fast = False):
	"""
	Computes the approximate pseudoinverse of any matrix using Cholesky Decomposition
	This means X @ pinv(X) approx = eye(n).
	
	Note that this is super fast, and will be used in HyperLearn as the default
	pseudoinverse solver for any matrix. Care is taken to make the algorithm
	converge, and this is done via forced epsilon regularization.
	
	HyperLearn's implementation also handles rank deficient and ill-conditioned
	matrices perfectly with the help of the limiting behaivour of
	adding forced epsilon regularization.
	
	Speed
	--------------
	If USE_GPU:
		Uses PyTorch's Cholesky. Speed is OK.
	If CPU:
		Uses Numpy's Fortran C based Cholesky.
		If NUMBA is not installed, uses very fast LAPACK functions.
	
	Stability
	--------------
	Alpha is added for regularization purposes. This prevents system
	rounding errors and promises better convergence rates.
	"""
	n,p = X.shape
	XT = X.T
	memoryCovariance(X)
	covariance = _XTX(XT) if n >= p else _XXT(XT)
	
	cho = cholesky(covariance, alpha = alpha, fast = fast)
	inv = invCholesky(cho, fast = fast)
	
	return inv @ XT if n >= p else XT @ inv



def svd(X, fast = True, U_decision = False, transpose = True):
	"""
	[Edited 9/11/2018 --> Modern Big Data Algorithms p/n ratio check]
	Computes the Singular Value Decomposition of any matrix.
	So, X = U * S @ VT. Note will compute svd(X.T) if p > n.
	Should be 99% same result. This means this implementation's
	time complexity is O[ min(np^2, n^2p) ]
	
	Speed
	--------------
	If USE_GPU:
		Uses PyTorch's SVD. PyTorch uses (for now) a NON divide-n-conquer algo.
		Submitted report to PyTorch:
		https://github.com/pytorch/pytorch/issues/11174
	If CPU:
		Uses Numpy's Fortran C based SVD.
		If NUMBA is not installed, uses divide-n-conqeur LAPACK functions.
	If Transpose:
		Will compute if possible svd(X.T) instead of svd(X) if p > n.
		Default setting is TRUE to maintain speed.
	
	Stability
	--------------
	SVD_Flip is used for deterministic output. Does NOT follow Sklearn convention.
	This flips the signs of U and VT, using VT_based decision.
	"""
	transpose = True if (transpose and X.shape[1] > X.shape[0]) else False
	X = _float(X)
	if transpose: 
		X, U_decision = X.T, not U_decision

	n, p = X.shape
	ratio = p/n
	#### TO DO: If memory usage exceeds LWORK, use GESVD
	if ratio >= 0.001:
		if USE_NUMBA:
			U, S, VT = numba.svd(X)
		else:
			#### TO DO: If memory usage exceeds LWORK, use GESVD
			U, S, VT, __ = lapack("gesdd", fast)(X, full_matrices = False)
	else:
		U, S, VT = lapack("gesvd", fast)(X, full_matrices = False)
		
	U, VT = svd_flip(U, VT, U_decision = U_decision)
	
	if transpose:
		return VT.T, S, U.T
	return U, S, VT
		


def pinv(X, alpha = None, fast = True):
	"""
	Computes the pseudoinverse of any matrix.
	This means X @ pinv(X) = eye(n).
	
	Optional alpha is used for regularization purposes.
	
	Speed
	--------------
	If USE_GPU:
		Uses PyTorch's SVD. PyTorch uses (for now) a NON divide-n-conquer algo.
		Submitted report to PyTorch:
		https://github.com/pytorch/pytorch/issues/11174
	If CPU:
		Uses Numpy's Fortran C based SVD.
		If NUMBA is not installed, uses divide-n-conqeur LAPACK functions.
	
	Stability
	--------------
	Condition number is:
		float32 = 1e3 * eps * max(S)
		float64 = 1e6 * eps * max(S)
	"""
	if alpha is not None: assert alpha >= 0
	alpha = 0 if alpha is None else alpha
	
	U, S, VT = svd(X, fast = fast)
	U, S, VT = _svdCond(U, S, VT, alpha)
	return (VT.T * S) @ U.T
	


def eigh(XTX, alpha = None, fast = True, svd = False, positive = False, qr = False):
	"""
	Computes the Eigendecomposition of a Hermitian Matrix
	(Positive Symmetric Matrix).
	
	Note: Slips eigenvalues / eigenvectors with MAX first.
	Scipy convention is MIN first, but MAX first is SVD convention.
	
	Uses the fact that the matrix is special, and so time
	complexity is approximately reduced by 1/2 or more when
	compared to full SVD.

	If POSITIVE is True, then all negative eigenvalues will be set
	to zero, and return value will be VT and not V.

	If SVD is True, then eigenvalues will be square rooted as well.
	
	Speed
	--------------
	If USE_GPU:
		Uses PyTorch's EIGH. PyTorch uses (for now) a non divide-n-conquer algo.
	If CPU:
		Uses Numpy's Fortran C based EIGH.
		If NUMBA is not installed, uses very fast divide-n-conqeur LAPACK functions.
		Note Scipy's EIGH as of now is NON divide-n-conquer.
		Submitted report to Scipy:
		https://github.com/scipy/scipy/issues/9212
		
	Stability
	--------------
	Alpha is added for regularization purposes. This prevents system
	rounding errors and promises better convergence rates.

	Also uses eig_flip to flip the signs of the eigenvectors
	to ensure deterministic output.
	"""
	n,p = XTX.shape
	assert n == p
	error = 1
	alpha = ALPHA_DEFAULT if alpha is None else alpha
	old_alpha = 0
	
	decomp = lapack("syevd", fast, "eigh") if not qr else lapack("syevr", fast) 

	while error != 0:
		if PRINT_ALL_WARNINGS: 
			print('eigh Alpha = {}'.format(alpha))

		# Add epsilon jitter to diagonal. Note adding
		# np.eye(p)*alpha is slower and uses p^2 memory
		# whilst flattening uses only p memory.
		addDiagonal(XTX, alpha-old_alpha)
		try:
			output = decomp(XTX)
			if USE_NUMBA: 
				S2, V = output
				error = 0
			else: 
				S2, V, error = output
		except: pass
		if error != 0:
			old_alpha = alpha
			alpha *= 10

	addDiagonal(XTX, -alpha)
	S2, V = S2[::-1], eig_flip(V[:,::-1])

	if svd or positive: 
		S2[S2 < 0] = 0.0
		V = V.T
	if svd:
		S2 **= 0.5
	return S2, V


_svd = svd
def eig(X, alpha = None, fast = True, U_decision = False, svd = False, stable = False):
	"""
	[Edited 8/11/2018 Made QR-SVD even faster --> changed to n >= p from n >= 5/3p]
	Computes the Eigendecomposition of any matrix using either
	QR then SVD or just SVD. This produces much more stable solutions 
	that pure eigh(covariance), and thus will be necessary in some cases.

	If STABLE is True, then EIGH will be bypassed, and instead SVD or QR/SVD
	will be used instead. This is to guarantee stability, since EIGH
	uses epsilon jitter along the diagonal of the covariance matrix.
	
	Speed
	--------------
	If n >= 5/3 * p:
		Uses QR followed by SVD noticing that U is not needed.
		This means Q @ U is not required, reducing work.

		Note Sklearn's Incremental PCA was used for the constant
		5/3 [`Matrix Computations, Third Edition, G. Holub and C. 
		Van Loan, Chapter 5, section 5.4.4, pp 252-253.`]

	Else If n >= p:
		SVD is used, as QR would be slower.

	Else If n <= p:
		SVD Transpose is used svd(X.T)

	If stable is False:
		Eigh is used or SVD depending on the memory requirement.
		
	Stability
	--------------
	Eig is the most stable Eigendecomposition in HyperLearn. It
	surpasses the stability of Eigh, as no epsilon jitter is added,
	unless specified when stable = False.
	"""
	n,p = X.shape
	memCheck = memoryXTX(X)

	# Note when p >= n, EIGH will return incorrect results, and hence HyperLearn
	# will default to SVD or QR/SVD
	if stable or not memCheck or p > n:
		# From Daniel Han-Chen's Modern Big Data Algorithms --> if n is larger
		# than p, then use QR. Old is >= 5/3*p.
		if n >= p:
			# Q, R = qr(X)
			# U, S, VT = svd(R)
			# S, VT is kept.
			__, S, V = _svd( qr(X, R_only = True), fast = fast, U_decision = U_decision)
		else:
			# Force turn on transpose:
			# either computes svd(X) or svd(X.T)
			# whichever is faster. [p >= n --> svd(X.T)]
			__, S, V = _svd(X, transpose = True, fast = fast, U_decision = U_decision)
		if not svd:
			S **= 2
			V = V.T
	else:
		S, V = eigh(_XTX(X.T), fast = fast, alpha = alpha)
		if svd:
			S **= 0.5
			V = V.T
			
	return S, V



	
def pinvh(XTX, alpha = None, fast = True):
	"""
	Computes the pseudoinverse of a Hermitian Matrix
	(Positive Symmetric Matrix) using Eigendecomposition.
	
	Uses the fact that the matrix is special, and so time
	complexity is approximately reduced by 1/2 or more when
	compared to full SVD.
	
	Speed
	--------------
	If USE_GPU:
		Uses PyTorch's EIGH. PyTorch uses (for now) a non divide-n-conquer algo.
	If CPU:
		Uses Numpy's Fortran C based EIGH.
		If NUMBA is not installed, uses very fast divide-n-conqeur LAPACK functions.
		Note Scipy's EIGH as of now is NON divide-n-conquer.
		Submitted report to Scipy:
		https://github.com/scipy/scipy/issues/9212
	
	Stability
	--------------
	Condition number is:
		float32 = 1e3 * eps * max(abs(S))
		float64 = 1e6 * eps * max(abs(S))
		
	Alpha is added for regularization purposes. This prevents system
	rounding errors and promises better convergence rates.
	"""
	assert XTX.shape[0] == XTX.shape[1]

	S2, V = eigh(XTX, alpha = alpha, fast = fast)
	S2, V = _eighCond(S2, V)
	return (V / S2) @ V.T



def pinvEig(X, alpha = None, fast = True):
	"""
	Computes the approximate pseudoinverse of any matrix X
	using Eigendecomposition on the covariance matrix XTX or XXT
	
	Uses a special trick where:
		If n >= p: X^-1 approx = (XT @ X)^-1 @ XT
		If n < p:  X^-1 approx = XT @ (X @ XT)^-1
	
	Speed
	--------------
	If USE_GPU:
		Uses PyTorch's EIGH. PyTorch uses (for now) a non divide-n-conquer algo.
	If CPU:
		Uses Numpy's Fortran C based EIGH.
		If NUMBA is not installed, uses very fast divide-n-conqeur LAPACK functions.
		Note Scipy's EIGH as of now is NON divide-n-conquer.
		Submitted report to Scipy:
		https://github.com/scipy/scipy/issues/9212
	
	Stability
	--------------
	Condition number is:
		float32 = 1e3 * eps * max(abs(S))
		float64 = 1e6 * eps * max(abs(S))
		
	Alpha is added for regularization purposes. This prevents system
	rounding errors and promises better convergence rates.
	"""
	n,p = X.shape
	XT = X.T
	memoryCovariance(X)
	covariance = _XTX(XT) if n >= p else _XXT(XT)
	
	S2, V = eigh(covariance, alpha = alpha, fast = fast)
	S2, V = _eighCond(S2, V)
	inv = (V / S2) @ V.T

	return inv @ XT if n >= p else XT @ inv
