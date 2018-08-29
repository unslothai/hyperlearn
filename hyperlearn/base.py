from torch import from_numpy as _toTensor, einsum as t_einsum, \
                transpose as __transpose, Tensor as typeTensor, stack as __stack, \
                float32, int32
from functools import wraps    
from numpy import finfo as np_finfo, einsum as np_einsum, log as np_log, array as np_array
from numpy import float32 as np_float32, float64 as np_float64, int32 as np_int32, \
                    int64 as np_int64, bool_ as np_bool, uint8 as np_uint8, ndarray, \
                    round as np_round
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from inspect import isclass as isClass

use_numpy_einsum = True
print_all_warnings = True

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

def toTensor(X):
    if type(X) is typeTensor:
        return X
    if isDict(X):
        X = list(X)
    if isList(X):
        X = np_array(X)
    try:
        if X.dtype == np_bool:
            X = X.astype(np_uint8)
        return _toTensor(X)
    except:
        return X

def isTensor(X):
    return type(X) is typeTensor


def Tensor(*args):
    out = []
    for x in args:
        if isTensor(x):
            out.append(  x  )
        else:
            out.append(  toTensor(x)  )
    return out


def Numpy(*args):
    out = []
    if isList(args[0]):
        args = args[0]

    for x in args:
        if isTensor(x):
            out.append(  x.numpy()  )
        else:
            out.append(  x  )
    return out


def return_numpy(*args):
    result = Numpy(*args)
    if len(result) == 1:
        return result[0]
    else:
        return tuple(result)

"""
------------------------------------------------------------
Decorators:
	n2n (numpy -> pytorch -> numpy)
	n2t (numpy -> pytorch)
Updated 27/8/2018
------------------------------------------------------------
"""
def n2n(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if isClass(args[0]):
            returned = f(args[0], *Tensor(*args[1:]), **kwargs)
        else:
            returned = f(*Tensor(*args), **kwargs)
        
        if not isList(returned):
            returned = [returned]
        
        return return_numpy(*returned)
    return wrapper


def n2t(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if isClass(args[0]):
            returned = f(args[0], *Tensor(*args[1:]), **kwargs)
        else:
            returned = f(*Tensor(*args), **kwargs)
        return returned
    return wrapper


def einsum(notation, *args, tensor = False):
    if use_numpy_einsum:
        args = Numpy(*args)
        out = np_einsum(notation, *args)
    else:
        args = Tensor(*args)
        try:
            out = t_einsum(notation, *args)
        except:
            out = t_einsum(notation, args)
    if tensor:
        return toTensor(out)
    return out

"""
------------------------------------------------------------
Matrix Manipulation
Updated 28/8/2018
------------------------------------------------------------
"""
def T(X):
    if not isTensor(X):
        X = toTensor(X)
    if len(X.shape) == 1:
        return X.reshape(-1,1)
    return __transpose(X, 0, 1)


def ravel(y, X):
    if isTensor(y):
        return y.flatten().type( X.dtype )
    else:
        return toTensor(y.ravel().astype( X.dtype ))


def row(y):
    y = np_array(y).ravel()
    try:
        return toTensor(y).type( int32 )
    except:
        return y

def row_np(y):
    return np_array(y).ravel()


def constant(X):
    return X.item()


def eps(X):
    try:
        return np_finfo(dtype(X)).eps
    except:
        return np_finfo(np_float64).eps
    

def dtype(tensor):
    string = str(tensor.dtype)
    if 'float32' in string: return np_float32
    elif 'float64' in string: return np_float64
    elif 'int32' in string: return np_int32
    elif 'int64' in string: return np_int64
    else: return np_float32
    

def stack(*args):
    if isList(args[0]):
        args = args[0]
    toStack = Tensor(*args)
    return __stack(toStack)

"""
------------------------------------------------------------
EINSUM, Einstein Notation Summation
Updated 28/8/2018
------------------------------------------------------------
"""

def squareSum(X):
    return einsum('ij,ij->i', X, X )

def rowSum(X, Y = None):
    if Y is None:
        return einsum('ij->i',X)
    return einsum('ij,ij->i', X , Y )
