
from libc.math cimport fabs
from libc.stdlib cimport rand

import numpy as np
cimport numpy as np
np.import_array()
from cython.parallel import parallel, prange

ctypedef np.ndarray ARRAY
ctypedef (long long) LONG
ctypedef bint bool

ctypedef (unsigned char) UINT8
ctypedef (unsigned short) UINT16
ctypedef (unsigned int) UINT32
ctypedef (unsigned long long) UINT64

ctypedef char INT8
ctypedef short INT16
ctypedef int INT32
ctypedef (long long) INT64

cdef int cycle = 0x7FFF
# Because of floating point rounding, force the output to be "rounded down"
# by reducing the range of the maximum possible number.
cdef int divisor = 0x7FFF + 1


######
cdef (int, int, int) low_high (int left, int right) nogil:
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
cdef void _bool(UINT8[:] out, LONG size, UINT64 x) nogil:
    cdef LONG i, k, j
    cdef UINT64 shift
    cdef UINT8 y
    
    for i in range(size // 64 + 1):
        x = (16807 * x + 2531011)

        # Use XOR gate to invert some bits
        shift = 1
        for j in range(32):
            x ^= shift; shift <<= 2

        # Get individual bits
        shift = 1
        for j in range(64):
            y = (x & shift) == 0
            out[k] = y
            k += 1
            if k >= size: break
            shift <<= 1
        if k >= size: break

######
cdef void _uint8(UINT8[:] out, LONG size, UINT32 x, float mult, float shift) nogil:
    cdef LONG i = 0
    while i < size:
        x = (8121 * x + 12345) & cycle; out[i] = <UINT8> (x*mult + shift); i += 1

        if i > size: break
        x = (x^0x6EEE)&cycle; out[i] = <UINT8> (x*mult + shift); i += 1
        
        if i > size: break
        x = (x^0xABCD)&cycle; out[i] = <UINT8> (x*mult + shift); i += 1
        
        if i > size: break
        x = (x^0xCCCC)&cycle; out[i] = <UINT8> (x*mult + shift); i += 1

######
cdef void _uint16(UINT16[:] out, LONG size, UINT32 x, float mult, float shift) nogil:
    cdef LONG i = 0
    while i < size:
        x = (65793 * x + 28411) & cycle; out[i] = <UINT16> (x*mult + shift); i += 1

        if i > size: break
        x = (x^0x6EEE)&cycle; out[i] = <UINT16> (x*mult + shift); i += 1
        
        if i > size: break
        x = (x^0xABCD)&cycle; out[i] = <UINT16> (x*mult + shift); i += 1
        
        if i > size: break
        x = (x^0xCCCC)&cycle; out[i] = <UINT16> (x*mult + shift); i += 1

######
cdef void _uint32(UINT32[:] out, LONG size, UINT32 x, float mult, float shift) nogil:
    cdef LONG i = 0
    while i < size:
        x = (214013 * x + 2531011) & cycle; out[i] = <UINT32> (x*mult + shift); i += 1

        if i > size: break
        x = (x^0x6EEE)&cycle; out[i] = <UINT32> (x*mult + shift); i += 1
        
        if i > size: break
        x = (x^0xABCD)&cycle; out[i] = <UINT32> (x*mult + shift); i += 1
        
        if i > size: break
        x = (x^0xCCCC)&cycle; out[i] = <UINT32> (x*mult + shift); i += 1

######
cdef void _uint64(UINT64[:] out, LONG size, UINT64 x, double mult, double shift) nogil:
    cdef LONG i = 0
    while i < size:
        x = (214013 * x + 2531011) & cycle; out[i] = <UINT64> (x*mult + shift); i += 1

        if i > size: break
        x = (x^0x6EEE)&cycle; out[i] = <UINT64> (x*mult + shift); i += 1
        
        if i > size: break
        x = (x^0xABCD)&cycle; out[i] = <UINT64> (x*mult + shift); i += 1
        
        if i > size: break
        x = (x^0xCCCC)&cycle; out[i] = <UINT64> (x*mult + shift); i += 1

######
cdef void _int8(INT8[:] out, LONG size, UINT32 x, float mult, float shift) nogil:
    cdef LONG i = 0
    while i < size:
        x = (8121 * x + 12345) & cycle; out[i] = <INT8> (x*mult + shift); i += 1

        if i > size: break
        x = (x^0x6EEE)&cycle; out[i] = <INT8> (x*mult + shift); i += 1
        
        if i > size: break
        x = (x^0xABCD)&cycle; out[i] = <INT8> (x*mult + shift); i += 1
        
        if i > size: break
        x = (x^0xCCCC)&cycle; out[i] = <INT8> (x*mult + shift); i += 1

######
cdef void _int16(INT16[:] out, LONG size, UINT32 x, float mult, float shift) nogil:
    cdef LONG i = 0
    while i < size:
        x = (65793 * x + 28411) & cycle; out[i] = <INT16> (x*mult + shift); i += 1

        if i > size: break
        x = (x^0x6EEE)&cycle; out[i] = <INT16> (x*mult + shift); i += 1
        
        if i > size: break
        x = (x^0xABCD)&cycle; out[i] = <INT16> (x*mult + shift); i += 1
        
        if i > size: break
        x = (x^0xCCCC)&cycle; out[i] = <INT16> (x*mult + shift); i += 1

######
cdef void _int32(INT32[:] out, LONG size, UINT32 x, float mult, float shift) nogil:
    cdef LONG i = 0
    while i < size:
        x = (214013 * x + 2531011) & cycle; out[i] = <INT32> (x*mult + shift); i += 1

        if i > size: break
        x = (x^0x6EEE)&cycle; out[i] = <INT32> (x*mult + shift); i += 1
        
        if i > size: break
        x = (x^0xABCD)&cycle; out[i] = <INT32> (x*mult + shift); i += 1
        
        if i > size: break
        x = (x^0xCCCC)&cycle; out[i] = <INT32> (x*mult + shift); i += 1

######
cdef void _int64(INT64[:] out, LONG size, UINT64 x, double mult, double shift) nogil:
    cdef LONG i = 0
    while i < size:
        x = (214013 * x + 2531011) & cycle; out[i] = <INT64> (x*mult + shift); i += 1

        if i > size: break
        x = (x^0x6EEE)&cycle; out[i] = <INT64> (x*mult + shift); i += 1
        
        if i > size: break
        x = (x^0xABCD)&cycle; out[i] = <INT64> (x*mult + shift); i += 1
        
        if i > size: break
        x = (x^0xCCCC)&cycle; out[i] = <INT64> (x*mult + shift); i += 1

######
cdef void _float32(float[:] out, LONG size, UINT32 x, float mult, float shift) nogil:
    cdef LONG i = 0
    while i < size:
        x = (214013 * x + 2531011) & cycle; out[i] = <float> (x*mult + shift); i += 1

        if i > size: break
        x = (x^0x6EEE)&cycle; out[i] = <float> (x*mult + shift); i += 1
        
        if i > size: break
        x = (x^0xABCD)&cycle; out[i] = <float> (x*mult + shift); i += 1
        
        if i > size: break
        x = (x^0xCCCC)&cycle; out[i] = <float> (x*mult + shift); i += 1

######
cdef void _float64(double[:] out, LONG size, UINT64 x, double mult, double shift) nogil:
    cdef LONG i = 0
    while i < size:
        x = (214013 * x + 2531011) & cycle; out[i] = <double> (x*mult + shift); i += 1

        if i > size: break
        x = (x^0x6EEE)&cycle; out[i] = <double> (x*mult + shift); i += 1
        
        if i > size: break
        x = (x^0xABCD)&cycle; out[i] = <double> (x*mult + shift); i += 1
        
        if i > size: break
        x = (x^0xCCCC)&cycle; out[i] = <double> (x*mult + shift); i += 1


######
cpdef ARRAY bool_(size = 10, int seed = -1):
    """
    Fast creation of boolean array. Uses LCG (Linear Congruential Generator)
    combined with some bit shifting & XOR gate.
    """
    cdef UINT8[::1] out
    cdef UINT8[:,::1] out2D
    cdef LONG i, n, p

    cdef UINT64 x = <UINT64> (rand() if seed < 0 else seed)
    cdef UINT64 shift = <UINT64> (16807*x + 2531011)
    cdef bool isTuple = type(size) is tuple

    if isTuple:
        n, p, out2D = size[0], size[1], np.zeros(size, dtype = np.uint8)
        with nogil, parallel():
            for i in prange(n): 
                _bool(out2D[i], p, shift + 97*<UINT32>i)
            _bool(out2D[:,0], p, x)
    else:
        out = np.zeros(size, dtype = np.uint8)
        _bool(out, size, x)
    return np.asarray(out2D).view(np.bool_) if isTuple else np.asarray(out).view(np.bool_)


######
cpdef ARRAY uint8_(int left = 0, int right = 10, size = 10, int seed = -1):
    cdef UINT8[::1] out
    cdef UINT8[:,::1] out2D
    cdef LONG i, n, p

    cdef UINT32 x = <UINT32> (rand() if seed < 0 else seed)
    cdef float mult = <float> ((right-left) if left > 0 else right)/divisor
    cdef UINT32 change = <UINT32> (8121 * x + 12345)
    cdef bool isTuple = type(size) is tuple

    if isTuple:
        n, p, out2D = size[0], size[1], np.zeros(size, dtype = np.uint8)
        with nogil, parallel():
            for i in prange(n): 
                _uint8(out2D[i], p, change + 97*<UINT32>i, mult, left)
    else:
        out = np.zeros(size, dtype = np.uint8)
        _uint8(out, size, x, mult, left)

    return np.asarray(out2D) if isTuple else np.asarray(out)


######
cpdef ARRAY uint16_(int left = 0, int right = 10, size = 10, int seed = -1):
    cdef UINT16[::1] out
    cdef UINT16[:,::1] out2D
    cdef LONG i, n, p

    cdef UINT32 x = <UINT32> (rand() if seed < 0 else seed)
    cdef float mult = <float> ((right-left) if left > 0 else right)/divisor
    cdef UINT32 change = <UINT32> (65793 * x + 28411)
    cdef bool isTuple = type(size) is tuple

    if isTuple:
        n, p, out2D = size[0], size[1], np.zeros(size, dtype = np.uint16)
        with nogil, parallel():
            for i in prange(n): 
                _uint16(out2D[i], p, change + 97*<UINT32>i, mult, left)
    else:
        out = np.zeros(size, dtype = np.uint16)
        _uint16(out, size, x, mult, left)

    return np.asarray(out2D) if isTuple else np.asarray(out)


######
cpdef ARRAY uint32_(int left = 0, int right = 10, size = 10, int seed = -1):
    cdef UINT32[::1] out
    cdef UINT32[:,::1] out2D
    cdef LONG i, n, p

    cdef UINT32 x = <UINT32> (rand() if seed < 0 else seed)
    cdef float mult = <float> ((right-left) if left > 0 else right)/divisor
    cdef UINT32 change = <UINT32> (214013 * x + 2531011)
    cdef bool isTuple = type(size) is tuple

    if isTuple:
        n, p, out2D = size[0], size[1], np.zeros(size, dtype = np.uint32)
        with nogil, parallel():
            for i in prange(n): 
                _uint32(out2D[i], p, change + 97*<UINT32>i, mult, left)
    else:
        out = np.zeros(size, dtype = np.uint32)
        _uint32(out, size, x, mult, left)

    return np.asarray(out2D) if isTuple else np.asarray(out)


######
cpdef ARRAY uint64_(int left = 0, int right = 10, size = 10, int seed = -1):
    cdef UINT64[::1] out
    cdef UINT64[:,::1] out2D
    cdef LONG i, n, p

    cdef UINT64 x = <UINT64> (rand() if seed < 0 else seed)
    cdef double mult = <double> ((right-left) if left > 0 else right)/divisor
    cdef UINT64 change = <UINT64> (214013 * x + 2531011)
    cdef bool isTuple = type(size) is tuple

    if isTuple:
        n, p, out2D = size[0], size[1], np.zeros(size, dtype = np.uint64)
        with nogil, parallel():
            for i in prange(n): 
                _uint64(out2D[i], p, change + 97*<UINT64>i, mult, left)
    else:
        out = np.zeros(size, dtype = np.uint64)
        _uint64(out, size, x, mult, left)

    return np.asarray(out2D) if isTuple else np.asarray(out)


######
cpdef ARRAY int8_(int left = 0, int right = 10, size = 10, int seed = -1):
    cdef INT8[::1] out
    cdef INT8[:,::1] out2D
    cdef LONG i, n, p
    cdef int low, high, add
    low, high, add = low_high(left, right)

    cdef float mult = <float> (high - low)/divisor
    cdef float shift = <float> (left if left >= 0 else (low - add))
    cdef UINT32 x = <UINT32> (rand() if seed < 0 else seed)
    cdef UINT32 change = <UINT32> (8121 * x + 12345)
    cdef bool isTuple = type(size) is tuple

    if isTuple:
        n, p, out2D = size[0], size[1], np.zeros(size, dtype = np.int8)
        with nogil, parallel():
            for i in prange(n): 
                _int8(out2D[i], p, change + 97*<UINT32>i, mult, shift)
    else:
        out = np.zeros(size, dtype = np.int8)
        _int8(out, size, x, mult, shift)

    return np.asarray(out2D) if isTuple else np.asarray(out)


######
cpdef ARRAY int16_(int left = 0, int right = 10, size = 10, int seed = -1):
    cdef INT16[::1] out
    cdef INT16[:,::1] out2D
    cdef LONG i, n, p
    cdef int low, high, add
    low, high, add = low_high(left, right)

    cdef float mult = <float> (high - low)/divisor
    cdef float shift = <float> (left if left >= 0 else (low - add))
    cdef UINT32 x = <UINT32> (rand() if seed < 0 else seed)
    cdef UINT32 change = <UINT32> (65793 * x + 28411)
    cdef bool isTuple = type(size) is tuple

    if isTuple:
        n, p, out2D = size[0], size[1], np.zeros(size, dtype = np.int16)
        with nogil, parallel():
            for i in prange(n): 
                _int16(out2D[i], p, change + 97*<UINT32>i, mult, shift)
    else:
        out = np.zeros(size, dtype = np.int16)
        _int16(out, size, x, mult, shift)

    return np.asarray(out2D) if isTuple else np.asarray(out)


######
cpdef ARRAY int32_(int left = 0, int right = 10, size = 10, int seed = -1):
    cdef INT32[::1] out
    cdef INT32[:,::1] out2D
    cdef LONG i, n, p
    cdef int low, high, add
    low, high, add = low_high(left, right)

    cdef float mult = <float> (high - low)/divisor
    cdef float shift = <float> (left if left >= 0 else (low - add))
    cdef UINT32 x = <UINT32> (rand() if seed < 0 else seed)
    cdef UINT32 change = <UINT32> (214013 * x + 2531011)
    cdef bool isTuple = type(size) is tuple

    if isTuple:
        n, p, out2D = size[0], size[1], np.zeros(size, dtype = np.int32)
        with nogil, parallel():
            for i in prange(n): 
                _int32(out2D[i], p, change + 97*<UINT32>i, mult, shift)
    else:
        out = np.zeros(size, dtype = np.int32)
        _int32(out, size, x, mult, shift)

    return np.asarray(out2D) if isTuple else np.asarray(out)


######
cpdef ARRAY int64_(int left = 0, int right = 10, size = 10, int seed = -1):
    cdef INT64[::1] out
    cdef INT64[:,::1] out2D
    cdef LONG i, n, p
    cdef int low, high, add
    low, high, add = low_high(left, right)

    cdef double mult = <double> (high - low)/divisor
    cdef double shift = <double> (left if left >= 0 else (low - add))
    cdef UINT64 x = <UINT64> (rand() if seed < 0 else seed)
    cdef UINT64 change = <UINT64> (214013 * x + 2531011)
    cdef bool isTuple = type(size) is tuple

    if isTuple:
        n, p, out2D = size[0], size[1], np.zeros(size, dtype = np.int64)
        with nogil, parallel():
            for i in prange(n): 
                _int64(out2D[i], p, change + 97*<UINT64>i, mult, shift)
    else:
        out = np.zeros(size, dtype = np.int64)
        _int64(out, size, x, mult, shift)

    return np.asarray(out2D) if isTuple else np.asarray(out)


######
cpdef ARRAY float32_(int left = 0, int right = 10, size = 10, int seed = -1):
    cdef float[::1] out
    cdef float[:,::1] out2D
    cdef LONG i, n, p
    cdef int low, high, add
    low, high, add = low_high(left, right)

    cdef float mult = <float> (high - low)/divisor
    cdef float shift = <float> (left if left >= 0 else (low - add))
    cdef UINT32 x = <UINT32> (rand() if seed < 0 else seed)
    cdef UINT32 change = <UINT32> (214013 * x + 2531011)
    cdef bool isTuple = type(size) is tuple

    if isTuple:
        n, p, out2D = size[0], size[1], np.zeros(size, dtype = np.float32)
        with nogil, parallel():
            for i in prange(n): 
                _float32(out2D[i], p, change + 97*<UINT32>i, mult, shift)
    else:
        out = np.zeros(size, dtype = np.float32)
        _float32(out, size, x, mult, shift)

    return np.asarray(out2D) if isTuple else np.asarray(out)


######
cpdef ARRAY float64_(int left = 0, int right = 10, size = 10, int seed = -1):
    cdef double[::1] out
    cdef double[:,::1] out2D
    cdef LONG i, n, p
    cdef int low, high, add
    low, high, add = low_high(left, right)

    cdef double mult = <double> (high - low)/divisor
    cdef double shift = <double> (left if left >= 0 else (low - add))
    cdef UINT64 x = <UINT64> (rand() if seed < 0 else seed)
    cdef UINT64 change = <UINT64> (214013 * x + 2531011)
    cdef bool isTuple = type(size) is tuple

    if isTuple:
        n, p, out2D = size[0], size[1], np.zeros(size, dtype = np.float64)
        with nogil, parallel():
            for i in prange(n): 
                _float64(out2D[i], p, change + 97*<UINT64>i, mult, shift)
    else:
        out = np.zeros(size, dtype = np.float64)
        _float64(out, size, x, mult, shift)

    return np.asarray(out2D) if isTuple else np.asarray(out)

