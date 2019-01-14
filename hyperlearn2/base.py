
from .cython.base import process
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

from .cython.base import lapack
"""
Calls LAPACK functions from Scipy. Use like this: lapack("LAPACK function")("args")
For example: lapack("gesdd")(X, full_matrices = False) for SVD. Types are automatically
determined from X, so no need to specify the type of the matrix.
>>> lapack(function, numba = None, turbo = True)
"""

from .cython.base import blas
"""
Calls BLAS functions from Scipy. Use like this: blas("BLAS function")("args")
For example: blas("syrk")(X) for symmetric matrix multiply. Types are automatically
determined from X, so no need to specify the type of the matrix.
>>> lapack(function, left = "")
"""

from .cython.base import isComplex
"""
Given a numpy datatype, returns if it's a complex type.
"""

from .cython.base import available_memory
"""
Returns the current memory in MB for a system.
"""

from .cython.base import isList

