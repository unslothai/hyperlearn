

#########
### Define globals
cdef double MAX_MEMORY = 0.94

#########
### Import libs
import numpy as np
cimport numpy as np
np.import_array()

from psutil import virtual_memory

from libc.stdlib cimport malloc, free
from cpython cimport bool as BOOL

#########
### New types
ctypedef np.ndarray ARRAY
ctypedef long long LONG
ctypedef np.dtype DTYPE
ctypedef bint bool

#########
### Internal variables
cdef char float32, float64, complex64, complex128, cfloat, ccomplex
float32, float64, complex64, complex128, cfloat, ccomplex = 102, 100, 70, 68, 103, 71
cdef char float16 = 101

cdef char bool_, int8, int16, int32, int64, cint, pointer
bool_, int8, int16, int32, int64, cint, pointer = 63, 98, 104, 108, 113, 105, 112

cdef char uint8, uint16, uint32, uint64, cuint, upointer
uint8, uint16, uint32, uint64, cuint, upointer = 66, 72, 76, 81, 73, 80

cdef char ERROR, PASS
ERROR, PASS = 1, 0


######### Memory check functions
###
cdef dict MEMORY_FUNCTIONS = {
    "full": full_, "extended": extended_, "same": same_,
    "triu": triu_, "squared": squared_, "columns": columns_,
    "extra": extra_, "truncated": truncated_, "minimum": minimum_,
    "min_left": min_left_, "min_right": min_right_
}

###
cdef LONG full_(LONG n, LONG p):
    cdef LONG out = n*p
    cdef LONG a = n*n
    cdef LONG b = p*p

    if a < b:   out += a
    else:       out += b
    return out

###
cdef LONG extended_(LONG n, LONG p):
    cdef LONG a = n*n
    cdef LONG b = p*p
    cdef LONG k = n if n < p else p
    cdef LONG out = n*p + k

    if a < b:   out += a
    else:       out += b
    return out

###
cdef LONG same_(LONG n, LONG p):
    return n*p

###
cdef LONG triu_(LONG n, LONG p):
    return p*p if p < n else n*p

###
cdef LONG squared_(LONG n, LONG p):
    return n*n

###
cdef LONG columns_(LONG n, LONG p):
    return p

###
cdef LONG extra_(LONG n, LONG p):
    cdef LONG a = n if n < p else p
    cdef LONG b = a*a
    a += b
    return a

###
cdef LONG truncated_(LONG n, LONG p, int k):
    cdef LONG a = n if n < p else p
    a += k + 1 + n + p
    a *= k
    return a

###
cdef LONG minimum_(LONG n, LONG p, int k):
    return k*(n + p + 1 + k)

###
cdef LONG min_left_(LONG n, LONG p, int k):
    return k*n

###
cdef LONG min_right_(LONG n, LONG p, int k):
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
    return <int> (<LONG> (<double> virtual_memory().available * MAX_MEMORY) >> 20)


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
    cdef int byte = dtype.itemsize
    cdef LONG a, b, multiplier
    cdef int need
    cdef checker = MEMORY_FUNCTIONS[memcheck]
    
    if len(shape) == 1:
        a = shape[0]
        shape = (1, a)
    try:
        multiplier = checker(*shape)
    except:
        a, b = shape[0], shape[1]
        multiplier = checker(a, b)
    return <int> (<LONG> (multiplier * byte) >> 20) # 10 == KB, 20 == MB


#########
cdef (int, char) arg_process(x, bool square):
    """
    Checks if object is a matrix and checks the datatype.
    """
    cdef LONG a, b
    cdef tuple shape
    cdef char dtype, dt
    dtype, dt = 0, 0
    cdef type d = type(x)
    cdef str out
    
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
            
            elif dtype == bool_:        dt = float32
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
cpdef (void) wrapper(
    dict memcheck, bool square, bool fractional, int memory_length,
    list memory_keys, function_signature, function_args,
    list args, dict kwargs
    ):
    """
    Performs memory checks, data type checks and other checks.
    Cythonized to reduce time from approx 500us to now 9us or so.
    """

    cdef int number_args = len(function_args)
    
    cdef int n_args, n_kwargs, memoryNeed, memoryFree
    n_args, n_kwargs = len(args), len(kwargs)
    cdef int size = n_args + n_kwargs
    
    cdef bool ifCheck = True
    cdef bool hasK = False
    cdef bool hasComponents = False
    cdef str i
    cdef type Xdtype, Kdtype, dtype
    cdef DTYPE X_dtype
    cdef LONG n, p, a
    cdef int whereK, n_components, j, otherYes
    whereK, n_components, otherYes, j, memoryNeed = 0, 0, 0, 0, 0
    cdef double temp_components, default_n_components
    temp_components, default_n_components = 0, 0
    cdef ARRAY X
    cdef temp
    
    cdef bool *duplicate
    cdef char *new_dtypes
    cdef char dt
    cdef tuple shape
    
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
cpdef str lapack(char dtype, BOOL turbo, str function):
    cdef bool TURBO = <bool> turbo
    if dtype == float32 and TURBO:
        return f"_lapack.s{function}"
    elif dtype == float64 or not TURBO or dtype == cfloat:
        return f"_lapack.d{function}"
    elif dtype == complex64 or dtype == ccomplex:
        return f"_lapack.c{function}"
    else:
        return f"_lapack.z{function}"
    
    
#########
cpdef str blas(char dtype, str function, str left):
    if dtype == float32:
        return f"_blas.{left}s{function}"
    elif dtype == float64 or dtype == cfloat:
        return f"_blas.{left}d{function}"
    elif dtype == complex64 or dtype == ccomplex:
        return f"_blas.{left}c{function}"
    else:
        return f"_blas.{left}z{function}"



