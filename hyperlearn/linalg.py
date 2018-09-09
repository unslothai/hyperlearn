
from scipy.linalg import lapack, lu as _lu, qr as _qr
from scipy.sparse import linalg as sparse
from . import numba
from .base import *
from .utils import *
from numpy import random as _random, finfo

__all__ = ['cholesky', 'invCholesky', 'pinvCholesky',
			'svd', 'lu', 'qr', 'pinv', 'pinvh', 'eigh', 'pinvEig',
			'truncatedEigh', 'truncatedSVD']


def cholesky(X, alpha = None, fast = True):
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
	assert X.shape[0] == X.shape[1]
	check = 1
	alpha = ALPHA_DEFAULT if alpha is None else alpha
	old_alpha = 0
	
	if USE_NUMBA: 
		while check != 0:
			if PRINT_ALL_WARNINGS: 
				print('Alpha = {}'.format(alpha))
			try:
				X.flat[::X.shape[0]+1] += (alpha - old_alpha)
				cho = numba.cholesky(X)
				check = 0
			except:
				old_alpha = alpha
				alpha *= 10
		cho = cho.T
	else:
		decomp = lapack.dpotrf
		if fast: decomp = lapack.spotrf if X.dtype == np.float32 else lapack.dpotrf
			
		while check != 0:
			if PRINT_ALL_WARNINGS: 
				print('Alpha = {}'.format(alpha))
			X.flat[::X.shape[0]+1] += (alpha - old_alpha)
			cho, check = decomp(X)
			if check == 0: break
			old_alpha = alpha
			alpha *= 10
	
	X.flat[::X.shape[0]+1] -= alpha
	return cho



def lu(X):
	"""
	Computes the LU Decomposition of any matrix with pivoting.
	Uses Scipy. Will utilise LAPACK later.
	"""
	return _lu(X, permute_l = True, check_finite = False)


def qr(X):
	"""
	Computes the reduced QR Decomposition of any matrix.
	Uses optimized NUMBA QR if avaliable else use's Scipy's
	version.
	"""
	if USE_NUMBA: return numba.qr(X)
	return _qr(X, mode = 'economic', check_finite = False)



def invCholesky(X, fast = False):
	"""
	Computes the Inverse of a Hermitian Matrix
	(Positive Symmetric Matrix) after provided with Cholesky's
	Upper Triangular Matrix.
	
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
	inverse = lapack.dtrtri
	if fast: inverse = lapack.strtri if X.dtype == np.float32 else lapack.dtrtri
		
	choInv = inverse(X)[0]
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
	covariance = XT @ X if n >= p else X @ XT
	
	cho = cholesky(covariance, alpha = alpha, fast = fast)
	inv = invCholesky(cho, fast = fast)
	
	return inv @ XT if n >= p else XT @ inv



def svd(X, fast = True, U_decision = False, transpose = True):
	"""
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
	if transpose: 
		X, U_decision = X.T, ~U_decision

	if USE_NUMBA:
		U, S, VT = numba.svd(X)
	else:
		_svd = lapack.dgesdd
		if fast: _svd = lapack.sgesdd if X.dtype == np.float32 else lapack.dgesdd

		U, S, VT, __ = _svd(X, full_matrices = 0)
		
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
	
#     if USE_NUMBA:
#         return numba.pinv(X)
#     else:
	U, S, VT = svd(X, fast = fast)
	U, S, VT = _svdCond(U, S, VT, alpha)
	return (VT.T * S) @ U.T
	


def eigh(X, alpha = None, fast = True):
	"""
	Computes the Eigendecomposition of a Hermitian Matrix
	(Positive Symmetric Matrix).
	
	Note: Slips eigenvalues / eigenvectors with MAX first.
	Scipy convention is MIN first, but MAX first is SVD convention.
	
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
	Alpha is added for regularization purposes. This prevents system
	rounding errors and promises better convergence rates.

	Also uses eig_flip to flip the signs of the eigenvectors
	to ensure deterministic output.
	"""
	assert X.shape[0] == X.shape[1]
	check = 1
	alpha = ALPHA_DEFAULT if alpha is None else alpha
	old_alpha = 0
	
	if USE_NUMBA:
		while check != 0:
			if PRINT_ALL_WARNINGS: 
				print('Alpha = {}'.format(alpha))
			try:
				X.flat[::X.shape[0]+1] += (-old_alpha + alpha)
				S2, V = numba.eigh(X)
				check = 0
			except:
				old_alpha = alpha
				alpha *= 10
	else:
		eig = lapack.dsyevd
		if fast: eig = lapack.ssyevd if X.dtype == np.float32 else lapack.dsyevd
			
		while check != 0:
			X.flat[::X.shape[0]+1] += (-old_alpha + alpha)
			S2, V, check = eig(X)
			if check == 0: break
			old_alpha = alpha
			alpha *= 10
			if PRINT_ALL_WARNINGS: 
				print('Alpha = {}'.format(alpha))
			
	X.flat[::X.shape[0]+1] -= alpha
	return S2[::-1], eig_flip(V[:,::-1])


	
def pinvh(X, alpha = None, fast = True):
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
	assert X.shape[0] == X.shape[1]

	S2, V = eigh(X, alpha = alpha, fast = fast)
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
	covariance = XT @ X if n >= p else X @ XT
	
	S2, V = eigh(covariance, alpha = alpha, fast = fast)
	S2, V = _eighCond(S2, V)
	inv = (V / S2) @ V.T

	return inv @ XT if n >= p else XT @ inv



def truncatedEigh(XTX, n_components = 2, tol = None):
	"""
	Computes the Truncated Eigendecomposition of a Hermitian Matrix
	(positive definite). K = 2 for default.
	Return format is LARGEST eigenvalue first.
	
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
	v = _random.uniform(-1, 1, size = p ).astype(dtype)

	S2, V = sparse.eigsh(XTX, k = n_components, tol = tol, v0 = v)
	V = eig_flip(V)
	
	return S2[::-1], V[:,::-1]



def truncatedSVD(X, n_components = 2, tol = None):
	"""
	Computes the Truncated SVD of any matrix. K = 2 for default.
	Return format is LARGEST singular first first.
	
	Speed
	--------------
	Uses ARPACK from Scipy to compute the truncated decomp. Note that
	to make it slightly more stable and faster, follows Sklearn's
	random intialization from -1 -> 1.

	Also note tolerance is resolution(X), and NOT eps(X)

	Might switch to SPECTRA in the future.
	
	Stability
	--------------
	SVD FLIP is called to flip the VT signs for deterministic
	output. Note uses VT based decision and not U based decision.
	"""
	n, p = X.shape
	dtype = X.dtype
	
	if tol is None: tol = finfo(dtype).resolution
	v = _random.uniform(-1, 1, size = p ).astype(dtype)

	U, S, VT = sparse.svds(X, k = n_components, tol = tol, v0 = v)
	U, S, VT = U[:, ::-1], S[::-1], VT[::-1]
	U, VT = svd_flip(U, VT, U_decision = False)
	
	return U, S, VT


