
from scipy.linalg import lapack
from numpy import eye, finfo, abs as _abs, arange as _arange, sign as _sign, uint as _uint
from .numba import *
from .base import *

_condition = {'f': 1e3, 'd': 1e6}

__all__ = ['svd_flip', 'cholesky', 'invCholesky', 'pinvCholesky',
			'SVD', 'pinv', 'pinvh', 'eigh', 'pinvEig',
			'_SVDCond', '_eighCond']


def svd_flip(U, VT, U_decision = True):
	"""
	Flips the signs of U and VT for SVD / Eigendecomposition
	in order to force deterministic output.

	Follows Sklearn convention by looking at U's maximum in columns
	as default.
	"""
	if U_decision:
		max_abs_cols = _abs(U).argmax(0)
		signs = _sign(U[max_abs_cols, _arange(U.shape[1], dtype = _uint)])
		if U is not None: 
			U *= signs
		VT *= signs.reshape(-1,1)
	else:
		# rows of v, columns of u
		max_abs_rows = _abs(VT).argmax(1)
		signs = _sign(VT[_arange(len(VT), dtype = _uint), max_abs_rows])
		if U is not None: 
			U *= signs
		VT *= signs.reshape(-1,1)
	return U, VT


def _SVDCond(U, S, VT, alpha):
	t = S.dtype.char.lower()
	cond = (S > (_condition[t] * finfo(t).eps * S[0]))
	rank = cond.sum()
	
	S /= (S**2 + alpha)
	return U[:, :rank], S[:rank], VT[:rank]


def _eighCond(S2, V):
	t = S2.dtype.char.lower()
	absS = _abs(S2)
	cond = (absS > (_condition[t] * finfo(t).eps * absS.max()) )
	S2 = S2[cond]
	
	return S2, V[:, cond]



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
			print('Alpha = {}'.format(alpha))
			try:
				X.flat[::X.shape[0]+1] += (-old_alpha + alpha)
				cho = numba_cholesky(X)
				check = 0
			except:
				old_alpha = alpha
				alpha *= 10
		cho = cho.T
	else:
		decomp = lapack.dpotrf
		if fast: decomp = lapack.spotrf if X.dtype == np.float32 else lapack.dpotrf
			
		while check != 0:
			print('Alpha = {}'.format(alpha))
			X.flat[::X.shape[0]+1] += (-old_alpha + alpha)
			cho, check = decomp(X)
			if check == 0: break
			old_alpha = alpha
			alpha *= 10
	
	X.flat[::X.shape[0]+1] -= alpha
	return cho


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


def SVD(X, fast = True):
	"""
	Computes the Singular Value Decomposition of any matrix.
	So, X = U * S @ VT
	
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
	SVD_Flip is used for deterministic output. Follows Sklearn convention.
	This flips the signs of U and VT.
	"""
	if USE_NUMBA:
		U, S, VT = numba_svd(X)
	else:
		svd = lapack.dgesdd
		if fast: svd = lapack.sgesdd if X.dtype == np.float32 else lapack.dgesdd

		U, S, VT, __ = svd(X, full_matrices = 0)
		
	U, VT = svd_flip(U, VT)
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
	U, S, VT = SVD(X, fast = fast)
	U, S, VT = _SVDCond(U, S, VT, alpha)
	return (VT.T * S) @ U.T
	


def eigh(X, alpha = None, fast = True):
	"""
	Computes the Eigendecomposition of a Hermitian Matrix
	(Positive Symmetric Matrix).
	
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
	"""
	assert X.shape[0] == X.shape[1]
	check = 1
	alpha = ALPHA_DEFAULT if alpha is None else alpha
	old_alpha = 0
	
	if USE_NUMBA:
		while check != 0:
			print('Alpha = {}'.format(alpha))
			try:
				X.flat[::X.shape[0]+1] += (-old_alpha + alpha)
				S2, V = numba_eigh(X)
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
			print('Alpha = {}'.format(alpha))
			
	#V = svd_flip(U = None, VT, U_decision = False)
	X.flat[::X.shape[0]+1] -= alpha
	return S2, V

	
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

