
from scipy.linalg import lapack as _lapack, blas as _blas
from .numba import funcs as _numba
#from .cfuncs import process
from .cfuncs import wrapper as _wrapper, lapack as __lapack, blas as __blas
from inspect import signature
from functools import wraps

#####
def process(f = None, memcheck = None, square = False, fractional = True):
    """
    [Added 14/11/18] [Edited 18/11/18 for speed]
    [Edited 25/11/18 Array deprecation of unicode]
    [Edited 27/11/18 Allows support for n_components]
    [Edited 4/12/18 Supports Fast JLT methods]
    [Edited 28/12/18 Cythonized. Speed up of 50 or more.]
    Decorator onto HyperLearn functions. Does 2 things:
    1. Convert datatypes to appropriate ones
    2. Convert matrices to arrays
    3. (Optional) checks memory bounds.

    input:      1 argument, 1 optional
    ----------------------------------------------------------
    f:          The function to be decorated
    memcheck:   lambda n,p: ... function or f(n,p)
    fractional: Whether to change float n_components to ints

    returns:    X arguments from function f
    ----------------------------------------------------------
    """
    if isinstance(memcheck, str):
        memcheck = {"X":memcheck}
    ###
    def decorate(f):
        # get function signature
        memory_length = len(memcheck)
        memory_keys = list(memcheck.keys())
        function_signature = signature(f)
        function_args = function_signature.parameters

        @wraps(f)
        def wrapper(*args, **kwargs):
            args = list(args)
            _wrapper(
                memcheck, square, fractional, memory_length,
                memory_keys, function_signature, function_args,
                args, kwargs
                ) # Process arguments via Cython
            # try execute function
            try:
                return f(*args, **kwargs)
            except:
                # Memory Error again --> didnt catch
                raise MemoryError("Operation requires more memory than "
        "what the system resources offer.")
        return wrapper

    if f:
        return decorate(f)
    return decorate


######
class lapack():
    """
    [Added 14/11/18] [Edited 28/12/18 Cythonized string part]
    Get a LAPACK function based on the dtype(X). Acts like Scipy's get lapack function.

    input:      1 argument, 2 optional
    ----------------------------------------------------------
    function:   String for lapack function eg: "getrf"
    turbo:      Boolean to indicate if float32 can be used.
    numba:      String for numba function.

    returns:    LAPACK or Numba function.
    ----------------------------------------------------------
    """
    def __init__(self, function, numba = None, turbo = True):
        self.function = function
        self.turbo = turbo
        self.f = None

        if numba is not  None:
            try: 
                self.f = eval(f'_numba.{numba}')
                self.function = numba
            except: pass

    def __call__(self, *args, **kwargs):
        if self.f == None:

            if len(args) > 0:
                dtype = args[0].dtype
            else:
                dtype = next(iter(kwargs.values())).dtype
            dtype = ord(dtype.char)

            self.f = eval(
                __lapack(dtype, self.turbo, self.function)
                )

        return self.f(*args, **kwargs)

###
class blas():
    """
    [Added 14/11/18]
    Get a BLAS function based on the dtype(X). Acts like Scipy's get blas function.

    input:      1 argument
    ----------------------------------------------------------
    function:   String for blas function eg: "getrf"

    returns:    BLAS function
    ----------------------------------------------------------
    """
    def __init__(self, function, left = ""):
        self.function = function
        self.f = None
        self.left = left

    def __call__(self, *args, **kwargs):
        if self.f == None:
            if len(args) > 0:
                dtype = args[0].dtype
            else:
                dtype = next(iter(kwargs.values())).dtype
            dtype = ord(dtype.char)
            
            self.f = eval(
                __blas(dtype, self.function, self.left)
                )

        return self.f(*args, **kwargs)

