
include "DEFINE.pyx"
#########
cdef double MAX_MEMORY = 0.94

from psutil import virtual_memory
from inspect import signature
from functools import wraps

from scipy.linalg import lapack as _lapack, blas as _blas
from ..numba import funcs as _numba


######### Memory check functions
###
cdef dict MEMORY_FUNCTIONS = {
    "full": full_, "extended": svd_, "same": same_,
    "triu": triu_, "squared": squared_, "columns": columns_,
    "extra": extra_, "truncated": truncated_, "minimum": minimum_,
    "min_left": min_left_, "min_right": min_right_
}

###
cdef SIZE full_(SIZE n, SIZE p):
    cdef:
        SIZE out = n*p
        SIZE a = n*n
        SIZE b = p*p

    if a < b:   out += a
    else:       out += b
    return out

###
cdef SIZE svd_(SIZE n, SIZE p):
    # LWORK calculation is incorrect. Use approximate heuristic
    # from actual checking of memory usage.
    # (MIN*MAX + MIN*MIN + MIN) for storing U, S, VT
    # (~ n*p + 1.5*MIN**2 + n + p) approximate for workspace
    # So = 2*MIN*MAX + 2.5*MIN**2 + 2*MIN + MAX
    # = 2*MIN(1 + MAX) + 2.5*MAX**2 + MAX
    cdef SIZE MIN, MAX
    if n < p:
        MIN, MAX = n, p
    else:
        MIN, MAX = p, n

    cdef SIZE usage = 2*MIN*(MAX + 1) + MAX
    usage += <SIZE>(2.5*<double>(MIN*MIN))
    return usage


###
cdef SIZE syevd_(SIZE n, SIZE p):
    # LWORK calculation is incorrect. Use approximate heuristic
    # from actual checking of memory usage.
    # (p + p**2) for output, storing W, V
    # (~ 2*p**2 + p) for approximate workspace
    # total = (2p + 3*p**2)
    cdef SIZE usage = 2*p + 3*p*p
    return usage


###
cdef SIZE syevr_(SIZE n, SIZE p):
    # LWORK calculation is incorrect. Use approximate heuristic
    # from actual checking of memory usage.
    # (p + p**2) for output, storing W, V
    # (~ p**2 + p) for approximate workspace
    # total = (2p + 2*p**2)
    cdef SIZE usage = 2*p + 2*p*p
    return usage


###
cdef SIZE potrf_(SIZE n, SIZE p):
    # LWORK calculation is incorrect. Use approximate heuristic
    # from actual checking of memory usage.
    # (p + p**2) for output, storing W, V
    # (~ p**2 + p) for approximate workspace
    # total = (2p + 2*p**2)
    cdef SIZE usage = 2*p + 2*p*p
    return usage


###
cdef SIZE same_(SIZE n, SIZE p):
    return n*p

###
cdef SIZE triu_(SIZE n, SIZE p):
    return p*p if p < n else n*p

###
cdef SIZE squared_(SIZE n, SIZE p):
    return n*n

###
cdef SIZE columns_(SIZE n, SIZE p):
    return p

###
cdef SIZE extra_(SIZE n, SIZE p):
    cdef SIZE a = n if n < p else p
    cdef SIZE b = a*a
    a += b
    return a

###
cdef SIZE truncated_(SIZE n, SIZE p, int k):
    cdef SIZE a = n if n < p else p
    a += k + 1 + n + p
    a *= k
    return a

###
cdef SIZE minimum_(SIZE n, SIZE p, int k):
    return k*(n + p + 1 + k)

###
cdef SIZE min_left_(SIZE n, SIZE p, int k):
    return k*n

###
cdef SIZE min_right_(SIZE n, SIZE p, int k):
    return k*p


############### Processing Functions
#########
cpdef bool isList(x):
    cdef type dtype = type(x)
    return dtype is list or dtype is tuple


#########
cpdef int available_memory():
    """
    Returns the current memory in MB for a system.
    """
    return <int> (<SIZE> (<double> virtual_memory().available * MAX_MEMORY) >> 20)


#########
cpdef bool isComplex(DTYPE dtype):
    """
    Given a numpy datatype, returns if it's a complex type.
    """
    cdef char x = ord(dtype.char)
    
    if x == complex64:      return True
    elif x == complex128:   return True
    elif x == ccomplex:     return True
    return False


#########
cdef int memory(tuple shape, DTYPE dtype, str memcheck):
    """
    Calculates the memory usage of a numpy array if it gets casted
    to another datatype.
    """
    cdef:
        int byte = dtype.itemsize, need = 0
        SIZE a, b, multiplier
        checker = MEMORY_FUNCTIONS[memcheck]

        
    if len(shape) == 1:
        a = shape[0]
        shape = (1, a)
    try:
        multiplier = checker(*shape)
    except:
        a, b = shape[0], shape[1]
        multiplier = checker(a, b)
    return <int> (<SIZE> (multiplier * byte) >> 20) # 10 == KB, 20 == MB


#########
cdef (int, char) arg_process(x, bool square):
    """
    Checks if object is a matrix and checks the datatype.
    """
    cdef:
        SIZE a, b
        tuple shape
        char dtype = 0, dt = 0
        type d = type(x)
        str out
    

    if d == np.ndarray:
        shape = x.shape
        if len(shape) > 1:
            if square: 
                # matrix must be a square one (n == p)
                a, b = shape[0], shape[1]
                if a != b:
                    raise AssertionError(f"2D array is not square. Dimensions seen are ({shape}).")
            # if float:
            dtype, dt = ord(x.dtype.char), ord(x.dtype.char)
            
            if dtype == float32:        return 0, PASS
            elif dtype == float64:      return 0, PASS
            elif dtype == cfloat:       return 0, PASS
            elif dtype == complex64:    return 0, PASS
            elif dtype == complex128:   return 0, PASS
            elif dtype == ccomplex:     return 0, PASS
            elif dtype == float16:      dt = float32
            
            elif dtype == boolean:      dt = float32
            elif dtype == int8:         dt = float32
            elif dtype == int16:        dt = float32
            elif dtype == int32:        dt = float32
            elif dtype == int64:        dt = float64
            elif dtype == cint:         dt = float32
            elif dtype == pointer:      dt = float64
            
            elif dtype == uint8:        dt = float32
            elif dtype == uint16:       dt = float32
            elif dtype == uint32:       dt = float32
            elif dtype == uint64:       dt = float64
            elif dtype == cuint:        dt = float32
            elif dtype == upointer:     dt = float64

            else: raise TypeError(f"Data type of ({dtype}) is not a numerical type.") 
                
            out = chr(dt)
            return memory(shape, np.dtype(out), "same"), dt

    return 0, ERROR


#########
cdef void _wrapper(
    dict memcheck, bool square, bool fractional, int memory_length,
    list memory_keys, function_signature, function_args,
    list args, dict kwargs
    ):
    """
    Performs memory checks, data type checks and other checks.
    Cythonized to reduce time from approx 500us to now 9us or so.
    """
    cdef:
        int number_args = len(function_args), n_args = len(args), n_kwargs = len(kwargs)
        int memoryNeed = 0, memoryFree = 0
        int size = n_args + n_kwargs
    
        bool ifCheck = True,  hasK = False, hasComponents = False
        str i
        type Xdtype, Kdtype, dtype
        DTYPE X_dtype
        SIZE n, p, a
        int whereK = 0, n_components = 0, j = 0, otherYes = 0
        double temp_components = 0, default_n_components = 0
        ARRAY X
        temp
    
        bool *duplicate
        char *new_dtypes
        char dt
        tuple shape
    

    #########
    # If no need to check memory (save a few nanoseconds)
    if n_kwargs > 0:
        for i in kwargs.keys():
            if i == "nocheck":
                del kwargs[i]
                ifCheck = False
                break
    
    if ifCheck:
        if size == 0:
            raise IndexError("Function needs >= 1 function arguments.")

        if size > number_args:
            raise IndexError(f"Function has too many inputs. Only ({number_args}) is needed.")
        
        #########
        # if 1st is matrix
        temp = args[0]
        Xdtype = type(temp)
        
        if Xdtype is np.matrix:
            n = temp.shape[0]
            if n == 1:    X = temp.A1 # flatten down
            else:         X = temp.A
            args[0] = X
        
        #########
        # if not array
        elif Xdtype is not np.ndarray:
            raise IndexError("First argument is not a 2D array. Must be an array.")
        else:
            X = args[0]
        
        n, p = X.shape[0], X.shape[1]
        
        #########
        # check if n_components is in function:
        for i in function_args.keys():
            if i == "n_components":
                hasK = True
                hasComponents = True
                break
            whereK += 1
            
        #########
        # find what n_components is
        if n_args > whereK:
            temp = args[whereK]
            try:
                temp_components = <double> temp
            except:
                raise AssertionError(f"n_components = ({temp}) is not "
                        "the correct type. Must be int or float.")
            default_n_components = temp_components
            
            if temp_components < 1:
                a = n if n < p else p # min(n, p)
                temp_components *= a
                if temp_components < 1:    temp_components = 1
            if temp_components > p:  temp_components = p
            
            n_components = <int> temp_components
            # if n_components is changed:
            if n_components != default_n_components:
                args[whereK] = n_components
                
            hasK = False # don't need to check kwargs
        
        #########
        # check booleans and if an array is seen
        # for eg: L_only, Q_only
        duplicate = <bool*> malloc(sizeof(bool) * n_args)
        
        for j in range(n_args):
            temp = args[j]
            dtype = type(temp)

            if dtype is BOOL and memory_length > 0:
                duplicate[j] = True
                if j >= memory_length:
                    raise IndexError(f"Too many arguments. Argument ({temp}) is wrong.")
                i = memory_keys[j]
                kwargs[i] = temp
            else:
                duplicate[j] = False
        
        #########
        # check if arguments are indeed accepted
        for i in kwargs.keys():
            try:
                function_args[i]
            except:
                raise NameError(f"Argument ({i}) is not recognised in function. "
                    f"Function accepted signature is ({function_signature}).")
                
            temp = kwargs[i]
            dtype = type(temp)
            
            if dtype is BOOL and "only" in i:
                otherYes += <int> temp
            
            #########
            # if n_components seen
            if hasK and i == "n_components":
                try:
                    temp_components = <double> temp
                except:
                    raise AssertionError(f"n_components = ({temp}) is not "
                            "the correct type. Must be int or float.")
                default_n_components = temp_components

                if temp_components < 1:
                    a = n if n < p else p # min(n, p)
                    temp_components *= a
                    if temp_components < 1:    temp_components = 1
                if temp_components > p:  temp_components = p

                n_components = <int> temp_components
                # if n_components is changed:
                if n_components != default_n_components:
                    kwargs[i] = n_components

                hasK = False # don't need to check kwargs
        
        #########
        # if too many TRUEs or no TRUEs
        if otherYes == memory_length-1 or otherYes == 0:
            for i in kwargs.keys():
                temp = kwargs[i]
                dtype = type(temp)
                if dtype is BOOL and "only" in i:
                    kwargs[i] = False # set to all False
            kwargs["X"] = True # all true
        else:
            kwargs["X"] = False
        
        #########
        # if n_components not seen
        i = "n_components"
        if hasK:
            # add default n_components as per function
            try:    n_components = function_args[i].default
            except: n_components = 1
            kwargs[i] = n_components
            default_n_components = n_components

        #########
        # Now check data types of arrays
        n_kwargs = len(kwargs)
        new_dtypes = <char*> malloc(sizeof(char) * (n_args + n_kwargs) )
        
        # Go through ARGS
        for j in range(n_args):
            temp = args[j]
            if j != 0:
                dtype = type(temp)
                
                # Convert matrices
                if dtype is np.matrix:
                    a = temp.shape[0]
                    args[j] = temp.A1 if a == 1 else temp.A
                    temp = args[j]
            a, dt = arg_process(temp, square)
            
            # Get datatype(X)
            if j == 0:
                if dt == PASS:
                    X_dtype = X.dtype
                else:
                    X_dtype = np.dtype(chr(dt))
            memoryNeed += a
            new_dtypes[j] = dt
        
        # Go through KWARGS
        j = n_args
        for i in kwargs.keys():
            temp = kwargs[i]
            dtype = type(temp)
            
            # Convert matrices
            if dtype is np.matrix:
                a = temp.shape[0]
                kwargs[i] = temp.A1 if a == 1 else temp.A
                temp = kwargs[i]
            a, dt = arg_process(temp, square)
            
            # Get datatype(X)
            memoryNeed += a
            new_dtypes[j] = dt
            j += 1
            
        #########
        # Now fix n_components
        if n_components != 0:  shape = (n, p, n_components)
        else:                  shape = (n, p)
            
        for i in memory_keys:
            if kwargs[i]:
                memoryNeed += memory(shape, X_dtype, memcheck[i])
                break
        
        #########
        # confirm memory is enough for data conversion
        if memoryNeed > 0:
            memoryFree = available_memory()
            if memoryNeed > memoryFree:
                raise MemoryError(f"Operation requires {memoryNeed} MB,"
            f"but {memoryFree} MB is free, so an extra {memoryNeed-memoryFree} MB is required.")
                    
        #########
        # convert data dtypes
        for j in range(n_args):
            dt = new_dtypes[j]
            if dt != PASS and dt != ERROR:
                X_dtype = np.dtype(chr(dt))
                args[j] = args[j].astype(X_dtype)
        
        j = n_args
        for i in kwargs.keys():
            dt = new_dtypes[j]
            if dt != PASS and dt != ERROR:
                X_dtype = np.dtype(chr(dt))
                kwargs[i] = kwargs[i].astype(X_dtype)
            j += 1
            
        #########
        # clean up args so no duplicates are seen
        n_args -= 1
        while n_args > 0:
            if duplicate[n_args]:
                del args[n_args]
            n_args -= 1
        del kwargs["X"]  # no need for first argument
        
        #########
        # if default n_components:
        if fractional is False and hasComponents is True:
            kwargs["n_components"] = default_n_components
        
        #########
        free(duplicate)
        free(new_dtypes)
        


#########
cdef class process():
    """
    Cython equivalent of Hyperlearn's base.process, but this is much faster.
        >>> process(memcheck = {}, square = False, fractional = True)
    [Added 7/1/19]

    Parameters
    -----------
    memcheck:       Dictionary of memcheck arguments or 1 string. Used to check
                    whether the matrix X satifies the system's memory constraints.
    square:         Check whether the matrix X must be square.
    fractional:     (default = True). Whether to convert n_components float to int
                    (eg: 0.5 == 50%*p)
    Returns
    -----------
    Wrapped up function - can call like normal functions.
    """
    cdef SIZE memory_length
    cdef memcheck, function_signature, function_args
    cdef list memory_keys
    cdef bool square, fractional, not_processed
    
    def __init__(self, memcheck = {}, bool square = False, bool fractional = True):
        if isinstance(memcheck, str):
            self.memcheck = {"X":memcheck}
        else:
            self.memcheck = memcheck
            
        self.memory_length = len(self.memcheck)
        self.memory_keys = list(self.memcheck.keys())
        self.square = square
        self.fractional = fractional
        self.not_processed = True
    
    def __call__(self, f = None):
        if self.not_processed:
            self.function_signature = signature(f)
            self.function_args = self.function_signature.parameters
            self.not_processed = False
            
        def decorate(f):
            #@wraps(f)
            def wrapper(*args, **kwargs):
                args = list(args)
                _wrapper(
                    self.memcheck, self.square, self.fractional, self.memory_length,
                    self.memory_keys, self.function_signature, self.function_args,
                    args, kwargs
                    ) # Process arguments
                try:
                    return f(*args, **kwargs)
                except MemoryError:
                    # Memory Error again --> didnt catch
                    raise MemoryError("Operation requires more memory than "
            "what the system resources offer.")
            return wrapper
        if f: return decorate(f)
        return decorate



#########
cdef class lapack():
    """
    Calls LAPACK functions from Scipy. Use like this: lapack("LAPACK function")("args")
    For example: lapack("gesdd")(X, full_matrices = False) for SVD. Types are automatically
    determined from X, so no need to specify the type of the matrix.
        >>> lapack(function, numba = None, turbo = True)
    """
    cdef str function
    cdef bool turbo
    cdef f
    
    def __init__(self, str function, numba = None, bool turbo = True):
        self.function = function
        self.turbo = turbo
        self.f = None

        if numba is not None:
            try: 
                self.f = eval(f'_numba.{numba}')
                self.function = numba
            except: 
                pass


    def __call__(self, *args, **kwargs):
        cdef DTYPE dtype
        cdef char dt
        cdef str fx
        
        if self.f is None:

            if len(args) > 0:
                dtype = args[0].dtype
            else:
                dtype = next(iter(kwargs.values())).dtype
            dt = ord(dtype.char)
            
            # Compare dtype of first data matrix
            if dt == float32 and self.turbo:
                fx = f"_lapack.s{self.function}"
            elif dt == float64 or not self.turbo or dt == cfloat:
                fx = f"_lapack.d{self.function}"
            elif dt == complex64 or dtype == ccomplex:
                fx = f"_lapack.c{self.function}"
            else:
                fx = f"_lapack.z{self.function}"
            self.f = eval(fx)

        return self.f(*args, **kwargs)
    


#########
cdef class blas():
    """
    Calls BLAS functions from Scipy. Use like this: blas("BLAS function")("args")
    For example: blas("syrk")(X) for symmetric matrix multiply. Types are automatically
    determined from X, so no need to specify the type of the matrix.
        >>> lapack(function, left = "")
    """
    cdef str function, left
    cdef f

    def __init__(self, str function, str left = ""):
        self.function = function
        self.f = None
        self.left = left


    def __call__(self, *args, **kwargs):
        cdef DTYPE dtype
        cdef char dt
        cdef str fx

        if self.f is None:

            if len(args) > 0:
                dtype = args[0].dtype
            else:
                dtype = next(iter(kwargs.values())).dtype
            dt = ord(dtype.char)
            
            # Compare dtype of first data matrix
            if dt == float32:
                fx = f"_blas.{self.left}s{self.function}"
            elif dt == float64 or dt == cfloat:
                fx = f"_blas.{self.left}d{self.function}"
            elif dt == complex64 or dt == ccomplex:
                fx = f"_blas.{self.left}c{self.function}"
            else:
                fx = f"_blas.{self.left}z{self.function}"
            self.f = eval(fx)

        return self.f(*args, **kwargs)

