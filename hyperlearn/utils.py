
import numpy as np
from .numba import jit, prange, sign
from .base import blas
dtypes = (np.uint8, np.uint16, np.uint32, np.uint64)

###
def uint(i, array = True):
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
	if array:
		return np.empty(1, dtype = x)
	return x


###
def do_until_success(f, epsilon_f, size, overwrite = False, alpha = None, *args, **kwargs):
	if len(args) > 0:
		X = args[0]
	else:
		X = next(iter(kwargs.values()))
	old_alpha = 0
	error = 1
	while error != 0:
		epsilon_f(X, size, alpha - old_alpha)
		try:
			out = f(*args, **kwargs)
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
			print(alpha)

	if not overwrite:
		epsilon_f(X, size, -alpha)
	return out

###
def add_jitter(X, size, alpha):
	X.flat[::size] += alpha

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

###
def sign_max(X, axis = 0, n_jobs = 1):
	"""
	[Added 19/11/2018]
	Returns the sign of the maximum absolute value of an axis of X.
	Uses BLAS if n_jobs = 1. Uses Numba otherwise.

	input:		1 argument, 2 optional
	----------------------------------------------------------
	X: 			Matrix X to be processed. Must be 2D array.
	axis:		Default = 0. 0 = column-wise. 1 = row-wise.
	n_jobs:		Default = 1. Uses multiple CPUs if n*p > 20,000^2.

	returns: 	sign array of -1,1
	----------------------------------------------------------
	"""
	def amax(X, axis, n, p):
		idamax = blas("amax", "i")
		if axis == 1:
			indices = np.zeros(n, dtype = int)
			for i in range(n):
				indices[i] = idamax(X[i])
			return sign(X[range(n), indices])
		else:
			indices = np.zeros(p, dtype = int)
			for i in range(p):
				indices[i] = idamax(X[:,i])
			return sign(X[indices, range(p)])
	
	n, p = X.shape
	if n_jobs != 1:
		if n*p > 20000**2:
			@jit(parallel = True)        
			def amax_numb_0(X, n, p):
				indices = np.zeros(p, dtype = np.dtype('int'))
				for i in prange(p):
					_max = 0
					s = 1
					for j in range(n):
						Xji = X[j, i]
						a = abs(Xji)
						if a > _max:
							_max = a
							s = np.sign(Xji)
					indices[i] = s
				return indices

			@jit(parallel = True)
			def amax_numb_1(X, n, p):
				indices = np.zeros(n, dtype = np.dtype('int'))
				for i in prange(n):
					_max = 0
					s = 1
					Xi = X[i]
					for j in range(p):
						Xij = Xi[j]
						a = abs(Xij)
						if a > _max:
							_max = a
							s = np.sign(Xij)
					indices[i] = s
				return indices
			
			if axis == 0:
				return amax_numb_0(X, n, p, n_jobs = -1)
			return amax_numb_1(X, n, p, n_jobs = -1)
	return amax(X, axis, n, p)


###
def svd_flip(U = None, VT = None, U_decision = False, n_jobs = 1):
	"""
	[Added 19/11/2018] [Edited 21/11/2018 Uses BLAS]
	Flips the signs of U and VT from a SVD or eigendecomposition.
	Default opposite to Sklearn's U decision. HyperLearn uses
	the maximum of rows of VT.

	input:		2 argument, 1 optional
	----------------------------------------------------------
	U:			U matrix from SVD
	VT:			VT matrix (V transpose) from SVD
	U_decision:	Default = False. If True, uses max from U.

	returns: 	Nothing. Inplace updates U, VT
	----------------------------------------------------------
	"""
	if U_decision is None:
		return

	scal = blas("scal")
	if U is not None:
		if U_decision:
			signs = sign_max(U, 0, n_jobs = n_jobs)
		else:
			signs = sign_max(VT, 1, n_jobs = n_jobs)

		index = np.where(signs == -1)[0]
		
		for i in index:
			VT[i] = scal(a = VT[i], x = -1)
			U[:,i] = scal(a = U[:,i], x = -1)
	else:
		# Eig flip on eigenvectors
		signs = sign_max(VT, 0, n_jobs = n_jobs)
		index = np.where(signs == -1)[0]

		for i in index:
			VT[:,i] = scal(a = VT[:,i], x = -1)			


###
def svdCondition(U, S, VT, alpha = None):
	"""
	[Added 21/11/2018]
	Uses Scipy's SVD condition number calculation to improve pseudoinverse
	stability. Uses (1e3, 1e6) * eps(S) * S[0] as the condition number.
	Everythimg below cond is set to 0.

	input:		3 argument, 1 optional
	----------------------------------------------------------
	U:			U matrix from SVD
	S:			S diagonal array from SVD
	VT:			VT matrix (V transpose) from SVD
	alpha:		Default = None. Ridge regularization

	returns: 	U, S/(S+alpha), VT updated.
	----------------------------------------------------------
	"""
	dtype = S.dtype.char.lower()
	if dtype == "f":
		cond = 1000 #1e3
	else:
		cond = 1000000 #1e6
	eps = np.finfo(t).eps
	first = S[0]
	eps = eps*first*cond

	rank = S.size-1
	while rank > 0:
		if S[rank] >= eps: break
		rank -= 1
	rank += 1

	if rank != S.size:
		U, S, VT = U[:, :rank], S[:rank], VT[:rank]

	update = True
	if alpha != 1e-8:
		if alpha != None:
			S /= (S**2 + alpha)
			update = False

	if update:
		S = np.divide(1, S, out = S)

	return U, S, VT

