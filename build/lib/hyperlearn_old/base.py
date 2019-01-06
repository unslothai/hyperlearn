from torch import from_numpy as _Tensor, einsum as t_einsum, \
				transpose as __transpose, Tensor as typeTensor, stack as __stack, \
				float32, int32
from functools import wraps    
from numpy import finfo as np_finfo, einsum as np_einsum, log as np_log, array as np_array
from numpy import float32 as np_float32, float64 as np_float64, int32 as np_int32, \
					int64 as np_int64, bool_ as np_bool, uint8 as np_uint8, ndarray, \
					round as np_round
from numpy import newaxis
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from inspect import isclass as isClass
from torch import matmul as torch_dot, diag, ones
from numpy import diag as np_diag, ones as np_ones # dot as np_dot

USE_NUMPY_EINSUM = True
PRINT_ALL_WARNINGS = False
USE_GPU = False
ALPHA_DEFAULT = 0.00001
USE_NUMBA = True

"""
------------------------------------------------------------
Type Checks
Updated 27/8/2018
------------------------------------------------------------
"""
FloatInt = (float, int)
ListTuple = (list, tuple)
ArrayType = (np_array, ndarray)
KeysValues = ( type({}.keys()), type({}.values()), dict )

def isList(X):
	return type(X) in ListTuple

def isArray(X):
	return type(X) in ArrayType

def isIterable(X):
	if isArray(X):
		if len(X.shape) == 1: return True
	return isList(X)

def isDict(X):
	return type(X) in KeysValues

def array(X):
	if isDict(X):
		X = list(X)
	return np_array(X)

def Tensor(X):
	if type(X) is typeTensor:
		return X
	if isDict(X):
		X = list(X)
	if isList(X):
		X = np_array(X)
	try:
		if X.dtype == np_bool:
			X = X.astype(np_uint8)
		return _Tensor(X)
	except:
		return X

def isTensor(X):
	return type(X) is typeTensor


def Tensors(*args):
	out = []
	for x in args:
		if x is None: out.append(None)
		if isTensor(x):
			out.append(  x  )
		else:
			out.append(  Tensor(x)  )
	return out


def Numpy(*args):
	out = []
	if isList(args[0]):
		args = args[0]

	for x in args:
		if x is None: out.append(None)
		if isTensor(x):
			out.append(  x.numpy()  )
		else:
			out.append(  x  )
	return out


def return_numpy(args):
	args = [args] if not isList(args) else args
	result = Numpy(*args)
	if len(result) == 1:
		return result[0]
	else:
		return tuple(result)

def return_torch(args):
	args = [args] if not isList(args) else args
	result = Tensors(*args)
	if len(result) == 1:
		return result[0]
	else:
		return tuple(result)

"""
------------------------------------------------------------
Decorators:
	check
Updated 31/8/2018
------------------------------------------------------------
"""

def check(f):
	@wraps(f)
	def wrapper(*args, **kwargs):
		if USE_GPU:
			if isClass(args[0]):
				returned = f(args[0], *Tensors(*args[1:]), **kwargs)
			returned = f(*Tensors(*args), **kwargs)
			return return_torch(returned)

		returned = f(*args, **kwargs)
		return return_numpy(returned)
	return wrapper

"""
------------------------------------------------------------
Matrix Manipulation
	>>> Now can specify the backend either GPU or CPU
	>>> Note on CPU --> Numpy is considerably faster
		when X(n,p) p>>n
Updated 30/8/2018
------------------------------------------------------------
"""
#dot = torch_dot if USE_GPU else np_dot

def T(X):
	A = X.reshape(-1,1) if len(X.shape) == 1 else X
	if USE_GPU: return A.t()
	return A.T

def cast(X, dtype):
	if USE_GPU: return X.type(dtype)
	return X.astype(dtype)

#def ravel(X):
#	if USE_GPU: return X.unsqueeze(1)
#	return X.ravel()

def constant(X):
	if USE_GPU: return X.item()
	return X

def eps(X):
	try:
		return np_finfo(dtype(X)).eps
	except:
		return np_finfo(np_float64).eps

def resolution(X):
	try:
		return np_finfo(dtype(X)).resolution
	except:
		return np_finfo(np_float32).resolution
	

def dtype(tensor):
	string = str(tensor.dtype)
	if 'float32' in string: return np_float32
	elif 'float64' in string: return np_float64
	elif 'int32' in string: return np_int32
	elif 'int64' in string: return np_int64
	else: return np_float32
	

def stack(*args):
	if isList(args[0]): args = args[0]
	toStack = Tensors(*args)
	return __stack(toStack)

"""
------------------------------------------------------------
EINSUM, Einstein Notation Summation
Updated 28/8/2018
------------------------------------------------------------
"""
def einsum(notation, *args, tensor = False):
	if USE_NUMPY_EINSUM:
		args = Numpy(*args)
		out = np_einsum(notation, *args)
	else:
		args = Tensors(*args)
		try:
			out = t_einsum(notation, *args)
		except:
			out = t_einsum(notation, args)
	if tensor:
		return Tensor(out)
	return out


def squareSum(X):
	if len(X.shape) == 1:
		return einsum('i,i->', X, X)
	return einsum('ij,ij->i', X, X )


def rowSum(X, Y = None, transpose_a = False):
	if Y is None:
		return einsum('ij->i',X)
	if transpose_a:
		return einsum('ji,ij->i', X , Y )
	return einsum('ij,ij->i', X , Y )


def diagSum(X, Y, transpose_a = False):
	if transpose_a:
		return einsum('ji,ij->', X , Y )
	return einsum('ij,ij->', X , Y )


