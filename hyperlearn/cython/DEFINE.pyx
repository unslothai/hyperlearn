
cimport numpy as np
import numpy as np
np.import_array()
from cpython cimport bool as BOOL
from cython.parallel import parallel, prange
from libc.math cimport fabs
from libc.stdlib cimport malloc, free

ctypedef np.ndarray ARRAY
ctypedef (long long) LONG
ctypedef bint bool
ctypedef Py_ssize_t SIZE
ctypedef np.dtype DTYPE

ctypedef (unsigned char) UINT8
ctypedef (unsigned short) UINT16
ctypedef (unsigned int) UINT32
ctypedef (unsigned long long) UINT64

ctypedef char INT8
ctypedef short INT16
ctypedef int INT32
ctypedef (long long) INT64



cdef char float32, float64, complex64, complex128, cfloat, ccomplex
float32, float64, complex64, complex128, cfloat, ccomplex = 102, 100, 70, 68, 103, 71
cdef char float16 = 101

cdef char boolean, int8, int16, int32, int64, cint, pointer
boolean, int8, int16, int32, int64, cint, pointer = 63, 98, 104, 108, 113, 105, 112

cdef char uint8, uint16, uint32, uint64, cuint, upointer
uint8, uint16, uint32, uint64, cuint, upointer = 66, 72, 76, 81, 73, 80

cdef char ERROR, PASS
ERROR, PASS = 1, 0


cdef SIZE uint8_max = <SIZE> np.iinfo(np.uint8).max
cdef SIZE uint16_max = <SIZE> np.iinfo(np.uint16).max
cdef SIZE uint32_max = <SIZE> np.iinfo(np.uint32).max

cdef SIZE int8_max = <SIZE> np.iinfo(np.int8).max
cdef SIZE int16_max = <SIZE> np.iinfo(np.int16).max
cdef SIZE int32_max = <SIZE> np.iinfo(np.int32).max

cdef float float32_max = <float> np.finfo(np.float32).max
cdef double float64_max = <double> np.finfo(np.float64).max

