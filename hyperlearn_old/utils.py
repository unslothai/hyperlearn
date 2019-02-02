
from numpy import uint, newaxis, finfo, float32, float64, zeros
from .numba import sign, arange
from numba import njit, prange
from psutil import virtual_memory
from .exceptions import FutureExceedsMemory
from scipy.linalg.blas import dsyrk, ssyrk		# For XTX, XXT
from scipy.linalg import lapack as _lapack
from . import numba
from .base import USE_NUMBA

__all__ = ['lapack','svd_flip', 'eig_flip', '_svdCond', '_eighCond',
			'memoryXTX', 'memoryCovariance', 'memorySVD', '_float',
			'traceXTX', 'fastDot', '_XTX', '_XXT',
			'rowSum', 'rowSum_A','reflect', 
			'addDiagonal', 'setDiagonal']

_condition = {'f': 1e3, 'd': 1e6}


class lapack():
	"""
	[Added 11/11/2018] [Edited 13/11/2018 -> made into a class]
	[Edited 14/11/2018 -> fixed class]
	Get a LAPACK function based on the dtype(X). Acts like Scipy.
	"""
	def __init__(self, function, fast = True, numba = None):
		self.function = function
		self.fast = fast
		self.f = None

		if numba != None and USE_NUMBA:
			try: f = eval(f'numba.{function}')
			except: pass
			f = eval(f)
			self.f = f

	def __repr__(self):
		return f"Calls LAPACK or Numba function {self.function}"

	def __call__(self, *args, **kwargs):
		if self.f == None:
			self.f = f"_lapack.d{self.function}"
			
			if len(args) == 0:
				a = next(iter(kwargs.values()))
			else:
				a = args[0]
			
			if a.dtype == np.float32 and self.fast:
				self.f = f"_lapack.s{self.function}"
			self.f = eval(self.f)

		return self.f(*args, **kwargs)
		


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


def memorySVD(X):
	"""
	Computes the approximate memory usage of SVD(X) [transpose or not].
	How it's computed:
		X = U * S * VT
		U(n,p) * S(p) * VT(p,p)
		This means RAM usgae is np+p+p^2 approximately.
	### TODO: Divide N Conquer SVD vs old SVD
	"""
	n,p = X.shape
	if n > p: n,p = p,n
	free = virtual_memory().free * 0.95
	byte = 4 if '32' in str(X.dtype) else 8

	U = n*p
	S = p
	VT = p*p
	memUsage = (U+S+VT) * byte

	return memUsage < free



def _float(data):
	dtype = str(data.dtype)
	if 'f' not in dtype:
		if '64' in dtype:
			return data.astype(float64)
		return data.astype(float32)
	return data


@njit(fastmath = True, nogil = True, cache = True)
def traceXTX(X):
	"""
	[Edited 18/10/2018]
	One drawback of truncated algorithms is that they can't output the correct
	variance explained ratios, since the full eigenvalue decomp needs to be
	done. However, using linear algebra, trace(XT*X) = sum(eigenvalues).

	So, this function outputs the trace(XT*X) efficiently without computing
	explicitly XT*X.

	Changes --> now uses Numba which is approx 20% faster.
	"""
	n, p = X.shape
	s = 0
	for i in range(n):
		for j in range(p):
			xij = X[i,j]
			s += xij*xij
	return s



def fastDot(A, B, C):
	"""
	[Added 23/9/2018]
	[Updated 1/10/2018 Error in calculating which is faster]
	Computes a fast matrix multiplication of 3 matrices.
	Either performs (A @ B) @ C or A @ (B @ C) depending which is more
	efficient.
	"""
	size = A.shape
	n = size[0]
	p = size[1] if len(size) > 1 else 1
	
	size = B.shape
	k = size[1] if len(size) > 1 else 1
	
	size = C.shape
	d = size[1] if len(size) > 1 else 1
	
	# Forward (A @ B) @ C
	# p*k*n + k*d*n = kn(p+d)
	forward = k*n*(p+d)
	
	# Backward A @ (B @ C)
	# p*d*n + k*d*p = pd(n+k)
	backward = p*d*(n+k)
	
	if forward <= backward:
		return (A @ B) @ C
	return A @ (B @ C)

	

def _XTX(XT):
	"""
	[Added 30/9/2018]
	Computes XT @ X much faster than naive XT @ X.
	Notice XT @ X is symmetric, hence instead of doing the
	full matrix multiplication XT @ X which takes O(np^2) time,
	compute only the upper triangular which takes slightly
	less time and memory.
	"""
	if XT.dtype == float64:
		return dsyrk(1, XT, trans = 0).T
	return ssyrk(1, XT, trans = 0).T



def _XXT(XT):
	"""
	[Added 30/9/2018]
	Computes X @ XT much faster than naive X @ XT.
	Notice X @ XT is symmetric, hence instead of doing the
	full matrix multiplication X @ XT which takes O(pn^2) time,
	compute only the upper triangular which takes slightly
	less time and memory.
	"""
	if XT.dtype == float64:
		return dsyrk(1, XT, trans = 1).T
	return ssyrk(1, XT, trans = 1).T



@njit(fastmath = True, nogil = True, cache = True)
def rowSum_0(X, norm = False):
	"""
	[Added 17/10/2018]
	Computes rowSum**2 for dense matrix efficiently, instead of using einsum
	"""
	n, p = X.shape
	S = zeros(n, dtype = X.dtype)

	for i in range(n):
		s = 0
		Xi = X[i]
		for j in range(p):
			Xij = Xi[j]
			s += Xij*Xij
		S[i] = s
	if norm:
		S**=0.5
	return S


@njit(fastmath = True, nogil = True, cache = True)
def rowSum_A(X, norm = False):
	"""
	[Added 22/10/2018]
	Computes rowSum**2 for dense array efficiently, instead of using einsum
	"""
	n = len(X)
	s = 0
	for i in range(n):
		s += X[i]**2
	if norm:
		s **= 0.5
	return s


def rowSum(X, norm = False):
	"""
	[Added 22/10/2018]
	Combines rowSum for matrices and arrays.
	"""
	if len(X.shape) > 1:
		return rowSum_0(X, norm)
	return rowSum_A(X, norm)


def _reflect(X):
	"""
	See reflect(X, n_jobs = N) documentation.
	"""
	n = len(X)
	for i in prange(1, n):
		Xi = X[i]
		for j in range(i):
			X[j, i] = Xi[j]
	return X
reflect_single = njit(_reflect, fastmath = True, nogil = True, cache = True)
reflect_parallel = njit(_reflect, fastmath = True, nogil = True, parallel = True)


def reflect(X, n_jobs = 1):
	"""
	[Added 15/10/2018] [Edited 18/10/2018]
	Reflects lower triangular of matrix efficiently to upper.
	Notice much faster than say X += X.T or naive:
		for i in range(n):
			for j in range(i, n):
				X[i,j] = X[j,i]
	In fact, it is much faster to perform vertically:
		for i in range(1, n):
			Xi = X[i]
			for j in range(i):
				X[j,i] = Xi[j]
	The trick is to notice X[i], which reduces array access.
	"""
	X = reflect_parallel(X) if n_jobs != 1 else reflect_single(X)
	return X


def addDiagonal(X, c = 1):
	"""
	[Added 11/11/2018]
	Add c to diagonal of matrix
	"""
	n = X.shape[0]
	X.flat[::n+1] += c

def setDiagonal(X, c = 1):
	"""
	[Added 11/11/2018]
	Set c to diagonal of matrix
	"""
	n = X.shape[0]
	X.flat[::n+1] = c

