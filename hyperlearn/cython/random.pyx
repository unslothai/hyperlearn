
from libc.math cimport fabs 
import numpy as np
cimport numpy as np
np.import_array()

ctypedef (long long) LONG
ctypedef (unsigned char) UINT8
ctypedef (unsigned short) UINT16
ctypedef (unsigned int) UINT32
ctypedef (unsigned long long) UINT64

ctypedef char INT8
ctypedef short INT16
ctypedef int INT32
ctypedef (long long) INT64


######
cdef (int, int, int) low_high (int left, int right):
    """
    If (-10, 0) is seen, same as (0, 10) - 10. So add abs(left)
    (-100, -20) --> (0, 120) - 100
    """
    cdef int low, high
    cdef int add = <int> fabs(left)
    
    if left < 0:
        low = 0
        high = right + add
    else:
        low = left
        high = right
    return low, high, add


######
cpdef np.ndarray bool_(LONG size = 10, int seed = 0):
    """
    Fast creation of boolean array. Uses LCG (Linear Congruential Generator)
    combined with some bit shifting & XOR gate.
    """
    cdef UINT64 x = <UINT64> seed
    cdef UINT8[:] out = np.zeros(size, dtype = np.uint8)
    cdef UINT8 y = 0
    cdef UINT64 shift = 1
    cdef LONG k = 0
    cdef int i, j

    with nogil:
        for i in range(size // 64 + 1):
            x = (16807 * x + 2531011)
            
            # Use XOR gate to invert some bits
            shift = 1
            for j in range(32):
                x ^= shift
                shift <<= 2
            
            # Get individual bits
            shift = 1
            for j in range(64):
                y = (x & shift) == 0
                out[k] = y
                k += 1
                if k > size: break
                shift <<= 1
            if k > size: break
                
    return np.asarray(out).view(dtype = np.bool)


######
cpdef np.ndarray uint8(int left = 0, int right = 10, LONG size = 10, int seed = 0):
    cdef UINT32 x = <UINT32> seed
    cdef UINT8[:] out = np.zeros(size, dtype = np.uint8)
    cdef int diff = right - left
    cdef int i

    with nogil:
        if left == 0:
            for i in range(size):
                x = (8121 * x + 12345) & 0x7FFF
                out[i] = <UINT8> (x % right)
        else:
            for i in range(size):
                x = (8121 * x + 12345) & 0x7FFF
                out[i] = <UINT8> ((x - left) % diff + left)
    return np.asarray(out)


######
cpdef np.ndarray int8(int left = -10, int right = 10, LONG size = 10, int seed = 0):
    cdef UINT32 x = <UINT32> seed
    cdef INT8[:] out = np.zeros(size, dtype = np.int8)
    cdef int low, high, add
    low, high, add = low_high(left, right)
    cdef int diff = high - low
    cdef int i
    
    with nogil:
        for i in range(size):
            x = (8121 * x + 12345) & 0x7FFF
            out[i] = <INT8> (((x - low) % diff + low) - add)
    return np.asarray(out)


######
cpdef np.ndarray uint16(int left = 0, int right = 10, LONG size = 10, int seed = 0):
    cdef UINT32 x = <UINT32> seed
    cdef UINT16[:] out = np.zeros(size, dtype = np.uint16)
    cdef int diff = right - left
    cdef int i

    with nogil:
        if left == 0:
            for i in range(size):
                x = (65793 * x + 28411) & 0x7FFF
                out[i] = <UINT16> (x % right)
        else:
            for i in range(size):
                x = (65793 * x + 28411) & 0x7FFF
                out[i] = <UINT16> ((x - left) % diff + left)
    return np.asarray(out)


######
cpdef np.ndarray int16(int left = -10, int right = 10, LONG size = 10, int seed = 0):
    cdef UINT32 x = <UINT32> seed
    cdef INT16[:] out = np.zeros(size, dtype = np.int16)
    cdef int low, high, add
    low, high, add = low_high(left, right)
    cdef int diff = high - low
    cdef int i
    
    with nogil:
        for i in range(size):
            x = (65793 * x + 28411) & 0x7FFF
            out[i] = <INT16> (((x - low) % diff + low) - add)
    return np.asarray(out)


######
cpdef np.ndarray uint32(int left = 0, int right = 10, LONG size = 10, int seed = 0):
    cdef UINT32 x = <UINT32> seed
    cdef UINT32[:] out = np.zeros(size, dtype = np.uint32)
    cdef int diff = right - left
    cdef int i
    
    with nogil:
        if left == 0:
            for i in range(size):
                x = (214013 * x + 2531011) & 0x7FFF
                out[i] = <UINT32> (x % right)
        else:
            for i in range(size):
                x = (214013 * x + 2531011) & 0x7FFF
                out[i] = <UINT32> ((x - left) % diff + left)
    return np.asarray(out)


######
cpdef np.ndarray int32(int left = -10, int right = 10, LONG size = 10, int seed = 0):
    cdef UINT32 x = <UINT32> seed
    cdef INT32[:] out = np.zeros(size, dtype = np.int32)
    cdef int low, high, add
    low, high, add = low_high(left, right)
    cdef int diff = high - low
    cdef int i
    
    with nogil:
        for i in range(size):
            x = (214013 * x + 2531011) & 0x7FFF
            out[i] = <INT32> (((x - low) % diff + low) - add)
    return np.asarray(out)


######
cpdef np.ndarray uint64(int left = 0, int right = 10, LONG size = 10, int seed = 0):
    cdef UINT64 x = <UINT64> seed
    cdef UINT64[:] out = np.zeros(size, dtype = np.uint64)
    cdef int diff = right - left
    cdef int i

    with nogil:
        if left == 0:
            for i in range(size):
                x = (214013 * x + 2531011) & 0x7FFF
                out[i] = <UINT64> (x % right)
        else:
            for i in range(size):
                x = (214013 * x + 2531011) & 0x7FFF
                out[i] = <UINT64> ((x - left) % diff + left)
    return np.asarray(out)


######
cpdef np.ndarray int64(int left = -10, int right = 10, LONG size = 10, int seed = 0):
    cdef UINT64 x = <UINT64> seed
    cdef INT64[:] out = np.zeros(size, dtype = np.int64)
    cdef int low, high, add
    low, high, add = low_high(left, right)
    cdef int diff = high - low
    cdef int i
    
    with nogil:
        for i in range(size):
            x = (214013 * x + 2531011) & 0x7FFF
            out[i] = <INT64> (((x - low) % diff + low) - add)
    return np.asarray(out)


######
cpdef np.ndarray float32(int left = -10, int right = 10, LONG size = 10, int seed = 0):
    cdef UINT32 x = <UINT32> seed
    cdef float[:] out = np.zeros(size, dtype = np.float32)
    cdef int low, high, add
    low, high, add = low_high(left, right)
    cdef int diff = high - low
    cdef int i
    cdef float mult = diff/0x7FFF
    cdef float shift
    
    if left == 0:  shift = <float> low
    else:          shift = <float> (low - add)
    
    with nogil:
        for i in range(size):
            x = (214013 * x + 2531011) & 0x7FFF
            out[i] = <float> (x*mult + shift)
    return np.asarray(out)


######
cpdef np.ndarray float64(int left = -10, int right = 10, LONG size = 10, int seed = 0):
    cdef UINT64 x = <UINT64> seed
    cdef double[:] out = np.zeros(size, dtype = np.float64)
    cdef int low, high, add
    low, high, add = low_high(left, right)
    cdef int diff = high - low
    cdef int i
    cdef double mult = diff/0x7FFF
    cdef double shift
    
    if left == 0:  shift = <double> low
    else:          shift = <double> (low - add)
    
    with nogil:
        for i in range(size):
            x = (214013 * x + 2531011) & 0x7FFF
            out[i] = <double> (x*mult + shift)
    return np.asarray(out)

