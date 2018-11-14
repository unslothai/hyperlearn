
from functools import wraps
from psutil import virtual_memory
from numpy import float32, float64, complex, array, ndarray, matrix
from scipy.linalg import lapack as _lapack, blas as _blas
from . import numba
from numba import njit

dtypes = (float32, float64, complex)
array = (array, ndarray)

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
	if dtype == float32: byte = 4
	elif dtype == float64: byte = 8
	else: byte = 12
	
	multiplier = memcheck(*X.shape)
	if type(multiplier) == tuple:
		multiplier = multiplier[0]
	need = multiplier * byte
	surplus = free - need
	return need, surplus > 0

###
def process(f = None, memcheck = None):
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
		mem = True
		if type(x) == matrix:
			x = x.A
		if type(x) in array:
			if len(x.shape) > 1:
				d = x.dtype
				if d not in dtypes:
					dt = str(d)
					if '64' in dt:
						x = x.astype(np.float64)
					elif 'complex' not in dt:
						x = x.astype(float32)
				n, m = memory(x, d, memcheck)
		return x, n, m
	###
	def decorate(f):
		@wraps(f)
		def wrapper(*args, **kwargs):
			mem = True
			args = list(args)
			skip = False
			l = len(args)
			# First check *args list (eg: f(A, B))
			if l > 0:
				for i in range(l): 
					args[i], n, m = _float(args[i])
					mem &= m
					if not mem:
						skip = True
						break
			l = len(kwargs)
			# Check **kwargs: f(a = A, b = B)
			if l > 0 and not skip:
				for i in kwargs: 
					kwargs[i], n, m = _float(kwargs[i])
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
	fast:		Boolean to indicate if float32 can be used.
	numba:		String for numba function.

	returns: 	LAPACK or Numba function.
	----------------------------------------------------------
	"""
	def __init__(self, function, fast = True, numba = None):
		self.function = function
		self.fast = fast
		self.f = None

		if numba != None:
			try: f = eval(f'numba.{function}')
			except: pass
			f = eval(f)
			self.f = f

	def __repr__(self):
		return f"Calls function {self.function}"

	def __call__(self, *args, **kwargs):
		if self.f == None:
			self.f = f"_lapack.d{self.function}"
			
			if len(args) == 0:
				a = next(iter(kwargs.values()))
			else:
				a = args[0]
			
			if a.dtype == float32 and self.fast:
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

	def __repr__(self):
		return f"Calls function {self.function}"

	def __call__(self, *args, **kwargs):
		if self.f == None:
			self.f = f"_blas.d{self.function}"
			
			if len(args) == 0:
				a = next(iter(kwargs.values()))
			else:
				a = args[0]
			
			if a.dtype == float32:
				self.f = f"_blas.s{self.function}"
			self.f = eval(self.f)

		return self.f(*args, **kwargs)

###
def jit(f, parallel = False):
	"""
	[Added 14/11/2018]
	Decorator onto Numba NJIT compiled code.

	input:		1 argument, 1 optional
	----------------------------------------------------------
	function:	Function to decorate
	parallel:	If true, sets cache automatically to false

	returns: 	CPU Dispatcher function
	----------------------------------------------------------
	"""
	cache = False if parallel else True

	def decorate(f):
		f = njit(f, fastmath = True, nogil = True, parallel = parallel, cache = cache)
		@wraps(f)
		def wrapper(*args, **kwargs):
			return f(*args, **kwargs)
		return wrapper
	
	if f:
		return decorate(f)
	return decorate
