
from functools import wraps
from psutil import virtual_memory
import numpy as np
from scipy.linalg import lapack as _lapack, blas as _blas
from . import numba as _numba

dtypes = set((np.dtype(np.float32), np.dtype(np.float64), np.dtype(np.complex)))
array = set((np.array, np.ndarray))

###
def memory(X, dtype, memcheck):
	"""
	[Added 14/11/2018]
	Checks if an operation on a matrix is within memory bounds.

	input:		3 arguments
	----------------------------------------------------------
	X:			Input matrix
	dtype:		Input datatype(matrix)
	memcheck:	lambda n,p: ... function or f(n,p)

	returns: 	2 arguments
	----------------------------------------------------------
	need:		Total memory required for operation
	surplus:	Boolean - True means within memory bounds.
	"""
	if memcheck == None: return 1, True
	
	free = virtual_memory().free * 0.97
	if dtype == np.float32: byte = 4
	elif dtype == np.float64: byte = 8
	else: byte = 12
	
	multiplier = memcheck(*X.shape)
	if type(multiplier) == tuple:
		multiplier = multiplier[0]
	need = multiplier * byte
	surplus = free - need
	return need, surplus > 0

###
def process(f = None, memcheck = None, square = False):
	"""
	[Added 14/11/2018]
	Decorator onto HyperLearn functions. Does 2 things:
	1. Convert datatypes to appropriate ones
	2. Convert matrices to arrays
	3. (Optional) checks memory bounds.

	input:		1 argument, 1 optional
	----------------------------------------------------------
	f:			The function to be decorated
	memcheck:	lambda n,p: ... function or f(n,p)

	returns: 	X arguments from function f
	----------------------------------------------------------
	"""
	###
	def _float(x):
		m = True
		a = 0  # if array
		n = 0  # need memory
		if type(x) == np.matrix:
			x = x.A
		if type(x) in array:
			a = 1
			if len(x.shape) > 1:
				# if matrix has to be a square one:
				if square:
					assert x.shape[0] == x.shape[1]
				d = x.dtype
				if d not in dtypes:
					# cast dtype to correct one if not f32, f64, complex
					dt = str(d)
					if '64' in dt:
						x = x.astype(np.float64)
					elif 'complex' not in dt:
						x = x.astype(np.float32)
				n, m = memory(x, d, memcheck)
		return x, n, m, a
	###
	def decorate(f):
		@wraps(f)
		def wrapper(*args, **kwargs):
			l, L = len(args), len(kwargs)
			if l + L == 0:
				# No arguments seen
				raise IndexError("Function needs >= 1 function arguments")
			A = 0  # check how many arrays were seen in arguments
			mem, skip = True, False
			args = list(args)
			
			# First check *args list (eg: f(A, B))
			if l > 0:
				for i in range(l): 
					args[i], n, m, a = _float(args[i])
					A += a
					mem &= m  # memory must work for ALL arguments
					if not mem:
						skip = True
						break

			# Check **kwargs: f(a = A, b = B)
			if L > 0 and not skip:
				for i in kwargs: 
					kwargs[i], n, m, a = _float(kwargs[i])
					A += a
					mem &= m
					if not mem:
						skip = True
						break
			if not mem:
				# Compute memory missing and raise Error if not good.
				n = int(n) >> 20    # 10 == KB, 20 == MB
				free = int(virtual_memory().free * 0.97) >> 20
				u = n-free
				
				raise MemoryError(f"""Operation requires {n} MB, but {free} MB is free,
				so an extra {u} MB is required.""")
			if A == 0:
				# No array was present in arguments
				raise TypeError(f"No array or matrix was provided as input.")
			
			return f(*args, **kwargs)
		return wrapper
	
	if f:
		return decorate(f)
	return decorate
	
###
class lapack(object):
	"""
	[Added 14/11/2018]
	Get a LAPACK function based on the dtype(X). Acts like Scipy's get lapack function.

	input:		1 argument, 2 optional
	----------------------------------------------------------
	function:	String for lapack function eg: "getrf"
	turbo:		Boolean to indicate if float32 can be used.
	numba:		String for numba function.

	returns: 	LAPACK or Numba function.
	----------------------------------------------------------
	"""
	def __init__(self, function, numba = None, turbo = True):
		self.function = function
		self.turbo = turbo
		self.f = None
		self.numba = False

		if numba != None:
			try: 
				self.f = eval(f'_numba.{numba}')
				self.function = numba
				self.numba = True
			except: pass

	def __call__(self, *args, **kwargs):
		if self.f == None:
			self.f = f"_lapack.d{self.function}"
			
			if len(args) == 0:
				# get first item
				a = next(iter(kwargs.values()))
			else:
				a = args[0]
			
			if a.dtype == np.float32 and self.turbo:
				self.f = f"_lapack.s{self.function}"
			self.f = eval(self.f)

		return self.f(*args, **kwargs)

###
class blas(object):
	"""
	[Added 14/11/2018]
	Get a BLAS function based on the dtype(X). Acts like Scipy's get blas function.

	input:		1 argument
	----------------------------------------------------------
	function:	String for blas function eg: "getrf"

	returns: 	BLAS function
	----------------------------------------------------------
	"""
	def __init__(self, function):
		self.function = function
		self.f = None


	def __call__(self, *args, **kwargs):
		if self.f == None:
			self.f = f"_blas.d{self.function}"
			
			if len(args) == 0:
				# get first item
				a = next(iter(kwargs.values()))
			else:
				a = args[0]
			
			if a.dtype == np.float32:
				self.f = f"_blas.s{self.function}"
			self.f = eval(self.f)

		return self.f(*args, **kwargs)
