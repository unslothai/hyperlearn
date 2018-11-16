
from functools import wraps
from psutil import virtual_memory
import numpy as np
from scipy.linalg import lapack as _lapack, blas as _blas
from . import numba as _numba
from .numba import _min
from inspect import signature

MAX_MEMORY = 0.94
lists = set((tuple, list))
type_function = type(lambda f: 1)

memory_usage = {
	"full" : 	lambda n,p: n*p + _min(n**2, p**2),
	"same" : 	lambda n,p: n*p,
	"triu" : 	lambda n,p: p**2 if p < n else n*p,
	"squared" :	lambda n,p: n**2,
	"columns" :	lambda n,p: p,
	}

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
	if memcheck == None: return 0
	
	if dtype == np.float32: byte = 4
	elif dtype in (np.float64, np.complex64): byte = 8
	else: byte = 12

	shape = X.shape
	if len(shape) == 1:
		shape = (1, shape[0])
	
	multiplier = memcheck(*shape)
	if type(multiplier) == tuple:
		multiplier = multiplier[0]
	need = (multiplier * byte) >> 20 # 10 == KB, 20 == MB

	# not enough memory, so show MemoryError
	# Compute memory missing and raise Error if not good.   
	free = int(virtual_memory().available * MAX_MEMORY) >> 20
	extra = need-free
	if need > free:
		raise MemoryError(f"""Operation requires {need} MB, but {free} MB is free,
	so an extra {extra} MB is required.""")
	return need

###
def arg_process(args, index, need, square):
	## confirm there are more than 1 matrices
	## and confirm if matrix is square if optional square argument is True
	## also check if matrix is type int, uint, and check memory first.
	dt = None # final dtype conversion
	isArray = 0 # count number of arrays seen
	x = args[index]
	if type(x) == np.matrix:
		if x.shape[0] == 1:
			args[index] = x.A1 # flatten down
		else:
			args[index] = x.A
	x = args[index]
	if type(x) == np.ndarray:
		shape = x.shape
		if len(shape) > 1:
			if square:
				# matrix must be a square one (n == p)
				if shape[0] != shape[1]:
					raise AssertionError(f"2D array is not square. Dimensions seen are {shape}.")
			# if float:
			dtype, dt = x.dtype, x.dtype
			same = memory_usage["same"]
			if dtype == np.float16:
				# float16 is not supported on CPU
				dt = np.float32
			elif np.issubdtype(dtype, np.integer):
				# conversion is needed, so check memory
				dt = np.float64 if dtype in (np.uint64, np.int64) else np.float32
			elif np.issubdtype(dtype, np.complexfloating) or np.issubdtype(dtype, np.floating):
				pass
			else:
				# not any numerical data dtype
				raise TypeError(f"Data type of {dtype} is not a numerical type.")
			if dt != dtype:	
				need += memory(x, dt, same)
			isArray = 1
	return dt, isArray

###
def arg_memory(data, x, data_dtype, check, memory_size):
	# now check memory allocation for algorithm
	need = 0
	if memory_size > 0:
		if type(x) == bool:
			if x:
				need = memory(data, data_dtype, check)
		elif x is data:
			need = memory(data, data_dtype, check)
	return need

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
	# convert all memory arguments to function checks
	if memcheck != None:
		if type(memcheck) == str:
			memcheck = [memcheck]
		if type(memcheck) in lists:
			memcheck = list(memcheck)
			for i, m in enumerate(memcheck):
				if type(memcheck[i]) != type_function:
					try:
						memcheck[i] = memory_usage[m]
					except:
						memcheck[i] = memory_usage["full"]
	###
	def decorate(f):
		# get function signature
		layout = signature(f)
		function_kwargs = set(layout.parameters.keys())

		@wraps(f)
		def wrapper(*args, **kwargs):
			memory_size = len(memcheck)
			## confirm there exists >= 1 arguments
			l, L = len(args), len(kwargs)
			size = l + L
			if size == 0:
				# No arguments seen
				raise IndexError("Function needs >= 1 function arguments.")

			# determine if kwargs names are within the scope of the function
			length = len(function_kwargs)
			if size > length:
				raise IndexError(f"Function has too many inputs. Only {length} is needed.")
			for x in kwargs:
				if x not in function_kwargs:
					raise NameError(f"Argument '{x}' is not recognised in function. Function accepted signature is {layout}.")

			need = 0 # total memory needed
			arg = 0 # keep track of which argument
			new_datatypes = [] # convert old data with new dtypes
			data, data_type = None, None
			seenArray = 0 # needs to be >= 1
			if l > 0:
				for i in range(l):
					dt, isArray = arg_process(args, i, need, square)
					new_datatypes.append(dt)
					seenArray += isArray

					if arg == 0:
						data = args[i]
						data_type = new_datatypes[-1]
						arg = 1

			if L > 0:
				for x in kwargs:
					dt, isArray = arg_process(kwargs, x, need, square)
					new_datatypes.append(dt)
					seenArray += isArray

					if arg == 0:
						data = kwargs[x]
						data_type = new_datatypes[-1]
						arg = 1

			# if no arrays seen, raise error
			if seenArray == 0:
				raise IndexError("Function needs >= 1 2D arrays as input. Now, 0 is seen.")
			# also, first item must be an array:
			if data_type is None:
				raise IndexError("First argument is not an array. Must be an array.")

			# check algorithm memory allocation:
			# first check if boolean arguments are provided beforehand.
			seenBool = False
			for i in range(l):
				a = args[i]
				if type(a) == bool:
					seenBool |= a
			for x in kwargs:
				a = kwargs[x]
				if type(a) == bool:
					seenBool |= a

			# now if seenBool is True, then skip first memcheck
			if seenBool:
				if l > 0:
					l -= 1
				elif L > 0:
					L -= 1
				arg = 1
				memory_size -= 1
			else:
				arg = 0

			# now check if memory can be allocated
			if memory_size > 0:
				if l > 0:
					for i in range(l):
						need += arg_memory(data, args[i], data_type, memcheck[arg], memory_size)
						memory_size -= 1
						arg += 1
						if memory_size == 0: break
				if L > 0 and memory_size > 0:
					for x in kwargs:
						need += arg_memory(data, kwargs[x], data_type, memcheck[arg], memory_size)
						memory_size -= 1
						arg += 1
						if memory_size == 0: break


			# check final total memory
			free = int(virtual_memory().available * MAX_MEMORY) >> 20
			extra = need-free
			if need > free:
				raise MemoryError(f"""Operation requires {need} MB, but {free} MB is free,
			so an extra {extra} MB is required.""")

			# since memory is good, change datatypes
			arg = 0
			if l > 0:
				for i in range(l):
					dt = new_datatypes[arg]
					if dt is None:
						args[i] = args[i].astype(dt)
						arg += 1

			if L > 0:
				for x in kwargs:
					dt = new_datatypes[arg]
					if dt is None:
						kwargs[x] = kwargs[x].astype(dt)
						arg += 1

			# finally execute function
			try:
				return f(*args, **kwargs)
			except MemoryError:
				# Memory Error again --> didnt catch
				raise MemoryError(f"""Operation requires more than {free} MB, which
				which more than system resources.""")
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
			elif a.dtype == np.complex64:
				self.f = f"_lapack.c{self.function}"
			elif a.dtype == np.complex128:
				self.f = f"_lapack.z{self.function}"
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
			elif a.dtype == np.complex64:
				self.f = f"_blas.c{self.function}"
			elif a.dtype == np.complex128:
				self.f = f"_blas.z{self.function}"
			self.f = eval(self.f)

		return self.f(*args, **kwargs)
