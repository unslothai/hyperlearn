
cimport numpy as np
import numpy as np
np.import_array()
from libc.math cimport fabs
from libc.stdlib cimport rand
from cython.parallel import parallel, prange

ctypedef np.ndarray ARRAY
ctypedef (long long) LONG
ctypedef bint bool

cdef char float32, float64, float16
float32, float64, float16 = 102, 100, 101

cdef char boolean, int8, int16, int32, int64, cint, pointer
boolean, int8, int16, int32, int64, cint, pointer = 63, 98, 104, 108, 113, 105, 112

cdef char uint8, uint16, uint32, uint64, cuint, upointer
uint8, uint16, uint32, uint64, cuint, upointer = 66, 72, 76, 81, 73, 80


cdef LONG uint8_max = <LONG> np.iinfo(np.uint8).max
cdef LONG uint16_max = <LONG> np.iinfo(np.uint16).max
cdef LONG uint32_max = <LONG> np.iinfo(np.uint32).max

cdef LONG int8_max = <LONG> np.iinfo(np.int8).max
cdef LONG int16_max = <LONG> np.iinfo(np.int16).max
cdef LONG int32_max = <LONG> np.iinfo(np.int32).max

cdef float float32_max = <float> np.finfo(np.float32).max
cdef double float64_max = <double> np.finfo(np.float64).max

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
    k = 0
    cdef UINT64 shift
    cdef LONG div = size // 64
    cdef int diff = size - div*64
    
    for i in range(div):
        x, shift = 16807 * x + 2531011, 1
        # Use XOR gate to invert some bits
        for j in range(32):
            x ^= shift; shift <<= 2
        # Get individual bits
        shift = 1
        for j in range(64):
            out[k] = ((x & shift) == 0); k += 1; shift <<= 1
            
    x, shift = 16807 * x + 2531011, 1
    for j in range(diff):
        out[k] = ((x & shift) == 0); k += 1; shift <<= 1

######
cdef void _uint8(
    UINT8[::1] out, LONG size, UINT32 x, float mult, float shift, LONG div, int diff) nogil:
    cdef LONG i, j
    j = 0
    for i in range(div):
        x = (8121 * x + 12345)&cycle; out[j] = <UINT8> (x*mult + shift); j+=1
        x = (x^0x6EEE)&cycle; out[j] = <UINT8> (x*mult + shift); j+=1
        x = (x^0xABCD)&cycle; out[j] = <UINT8> (x*mult + shift); j+=1
        x = (x^0xCCCC)&cycle; out[j] = <UINT8> (x*mult + shift); j+=1
    if diff >= 1:   x = (8121 * x + 12345)&cycle; out[j] = <UINT8> (x*mult + shift); j+=1
    if diff >= 2:   x = (x^0x6EEE)&cycle; out[j] = <UINT8> (x*mult + shift); j+=1
    if diff >= 3:   x = (x^0xABCD)&cycle; out[j] = <UINT8> (x*mult + shift); j+=1
        
######
cdef void _uint16(
    UINT16[::1] out, LONG size, UINT32 x, float mult, float shift, LONG div, int diff) nogil:
    cdef LONG i, j
    j = 0
    for i in range(div):
        x = (65793 * x + 28411)&cycle; out[j] = <UINT16> (x*mult + shift); j+=1
        x = (x^0x6EEE)&cycle; out[j] = <UINT16> (x*mult + shift); j+=1
        x = (x^0xABCD)&cycle; out[j] = <UINT16> (x*mult + shift); j+=1
        x = (x^0xCCCC)&cycle; out[j] = <UINT16> (x*mult + shift); j+=1
    if diff >= 1:   x = (65793 * x + 28411)&cycle; out[j] = <UINT16> (x*mult + shift); j+=1
    if diff >= 2:   x = (x^0x6EEE)&cycle; out[j] = <UINT16> (x*mult + shift); j+=1
    if diff >= 3:   x = (x^0xABCD)&cycle; out[j] = <UINT16> (x*mult + shift); j+=1

######
cdef void _uint32(
    UINT32[::1] out, LONG size, UINT32 x, float mult, float shift, LONG div, int diff) nogil:
    cdef LONG i, j
    j = 0
    for i in range(div):
        x = (214013 * x + 2531011)&cycle; out[j] = <UINT32> (x*mult + shift); j+=1
        x = (x^0x6EEE)&cycle; out[j] = <UINT32> (x*mult + shift); j+=1
        x = (x^0xABCD)&cycle; out[j] = <UINT32> (x*mult + shift); j+=1
        x = (x^0xCCCC)&cycle; out[j] = <UINT32> (x*mult + shift); j+=1
    if diff >= 1:   x = (214013 * x + 2531011)&cycle; out[j] = <UINT32> (x*mult + shift); j+=1
    if diff >= 2:   x = (x^0x6EEE)&cycle; out[j] = <UINT32> (x*mult + shift); j+=1
    if diff >= 3:   x = (x^0xABCD)&cycle; out[j] = <UINT32> (x*mult + shift); j+=1

######
cdef void _uint64(
    UINT64[::1] out, LONG size, UINT64 x, double mult, double shift, LONG div, int diff) nogil:
    cdef LONG i, j
    j = 0
    for i in range(div):
        x = (214013 * x + 2531011)&cycle; out[j] = <UINT64> (x*mult + shift); j+=1
        x = (x^0x6EEE)&cycle; out[j] = <UINT64> (x*mult + shift); j+=1
        x = (x^0xABCD)&cycle; out[j] = <UINT64> (x*mult + shift); j+=1
        x = (x^0xCCCC)&cycle; out[j] = <UINT64> (x*mult + shift); j+=1
    if diff >= 1:   x = (214013 * x + 2531011)&cycle; out[j] = <UINT64> (x*mult + shift); j+=1
    if diff >= 2:   x = (x^0x6EEE)&cycle; out[j] = <UINT64> (x*mult + shift); j+=1
    if diff >= 3:   x = (x^0xABCD)&cycle; out[j] = <UINT64> (x*mult + shift); j+=1

######
cdef void _int8(
    INT8[::1] out, LONG size, UINT32 x, float mult, float shift, LONG div, int diff) nogil:
    cdef LONG i, j
    j = 0
    for i in range(div):
        x = (8121 * x + 12345)&cycle; out[j] = <INT8> (x*mult + shift); j+=1
        x = (x^0x6EEE)&cycle; out[j] = <INT8> (x*mult + shift); j+=1
        x = (x^0xABCD)&cycle; out[j] = <INT8> (x*mult + shift); j+=1
        x = (x^0xCCCC)&cycle; out[j] = <INT8> (x*mult + shift); j+=1
    if diff >= 1:   x = (8121 * x + 12345)&cycle; out[j] = <INT8> (x*mult + shift); j+=1
    if diff >= 2:   x = (x^0x6EEE)&cycle; out[j] = <INT8> (x*mult + shift); j+=1
    if diff >= 3:   x = (x^0xABCD)&cycle; out[j] = <INT8> (x*mult + shift); j+=1

######
cdef void _int16(
    INT16[::1] out, LONG size, UINT32 x, float mult, float shift, LONG div, int diff) nogil:
    cdef LONG i, j
    j = 0
    for i in range(div):
        x = (65793 * x + 28411)&cycle; out[j] = <INT16> (x*mult + shift); j+=1
        x = (x^0x6EEE)&cycle; out[j] = <INT16> (x*mult + shift); j+=1
        x = (x^0xABCD)&cycle; out[j] = <INT16> (x*mult + shift); j+=1
        x = (x^0xCCCC)&cycle; out[j] = <INT16> (x*mult + shift); j+=1
    if diff >= 1:   x = (65793 * x + 28411)&cycle; out[j] = <INT16> (x*mult + shift); j+=1
    if diff >= 2:   x = (x^0x6EEE)&cycle; out[j] = <INT16> (x*mult + shift); j+=1
    if diff >= 3:   x = (x^0xABCD)&cycle; out[j] = <INT16> (x*mult + shift); j+=1

######
cdef void _int32(
    INT32[::1] out, LONG size, UINT32 x, float mult, float shift, LONG div, int diff) nogil:
    cdef LONG i, j
    j = 0
    for i in range(div):
        x = (214013 * x + 2531011)&cycle; out[j] = <INT32> (x*mult + shift); j+=1
        x = (x^0x6EEE)&cycle; out[j] = <INT32> (x*mult + shift); j+=1
        x = (x^0xABCD)&cycle; out[j] = <INT32> (x*mult + shift); j+=1
        x = (x^0xCCCC)&cycle; out[j] = <INT32> (x*mult + shift); j+=1
    if diff >= 1:   x = (214013 * x + 2531011)&cycle; out[j] = <INT32> (x*mult + shift); j+=1
    if diff >= 2:   x = (x^0x6EEE)&cycle; out[j] = <INT32> (x*mult + shift); j+=1
    if diff >= 3:   x = (x^0xABCD)&cycle; out[j] = <INT32> (x*mult + shift); j+=1

######
cdef void _int64(
    INT64[::1] out, LONG size, UINT64 x, double mult, double shift, LONG div, int diff) nogil:
    cdef LONG i, j
    j = 0
    for i in range(div):
        x = (214013 * x + 2531011)&cycle; out[j] = <INT64> (x*mult + shift); j+=1
        x = (x^0x6EEE)&cycle; out[j] = <INT64> (x*mult + shift); j+=1
        x = (x^0xABCD)&cycle; out[j] = <INT64> (x*mult + shift); j+=1
        x = (x^0xCCCC)&cycle; out[j] = <INT64> (x*mult + shift); j+=1
    if diff >= 1:   x = (214013 * x + 2531011)&cycle; out[j] = <INT64> (x*mult + shift); j+=1
    if diff >= 2:   x = (x^0x6EEE)&cycle; out[j] = <INT64> (x*mult + shift); j+=1
    if diff >= 3:   x = (x^0xABCD)&cycle; out[j] = <INT64> (x*mult + shift); j+=1

######
cdef void _float32(
    float[::1] out, LONG size, UINT32 x, float mult, float shift, LONG div, int diff) nogil:
    cdef LONG i, j
    j = 0
    for i in range(div):
        x = (214013 * x + 2531011)&cycle; out[j] = <float> (x*mult + shift); j+=1
        x = (x^0x6EEE)&cycle; out[j] = <float> (x*mult + shift); j+=1
        x = (x^0xABCD)&cycle; out[j] = <float> (x*mult + shift); j+=1
        x = (x^0xCCCC)&cycle; out[j] = <float> (x*mult + shift); j+=1
    if diff >= 1:   x = (214013 * x + 2531011)&cycle; out[j] = <float> (x*mult + shift); j+=1
    if diff >= 2:   x = (x^0x6EEE)&cycle; out[j] = <float> (x*mult + shift); j+=1
    if diff >= 3:   x = (x^0xABCD)&cycle; out[j] = <float> (x*mult + shift); j+=1

######
cdef void _float64(
    double[::1] out, LONG size, UINT64 x, double mult, double shift, LONG div, int diff) nogil:
    cdef LONG i, j
    j = 0
    for i in range(div):
        x = (214013 * x + 2531011)&cycle; out[j] = <double> (x*mult + shift); j+=1
        x = (x^0x6EEE)&cycle; out[j] = <double> (x*mult + shift); j+=1
        x = (x^0xABCD)&cycle; out[j] = <double> (x*mult + shift); j+=1
        x = (x^0xCCCC)&cycle; out[j] = <double> (x*mult + shift); j+=1
    if diff >= 1:   x = (214013 * x + 2531011)&cycle; out[j] = <double> (x*mult + shift); j+=1
    if diff >= 2:   x = (x^0x6EEE)&cycle; out[j] = <double> (x*mult + shift); j+=1
    if diff >= 3:   x = (x^0xABCD)&cycle; out[j] = <double> (x*mult + shift); j+=1


######
cdef ARRAY bool_(size = 10, int seed = -1):
    """
    Fast creation of boolean array. Uses LCG (Linear Congruential Generator)
    combined with some bit shifting & XOR gate.
    """
    cdef UINT8[::1] out
    cdef UINT8[:,:] out2D
    cdef LONG i, n, p

    cdef UINT64 x = <UINT64> (rand() if seed < 0 else seed)
    cdef UINT64 shift = <UINT64> (16807*x + 2531011)
    cdef bool isTuple = type(size) is tuple

    if isTuple:
        n, p, out2D = size[0], size[1], np.zeros(size, dtype = np.uint8)
        with nogil, parallel():
            for i in prange(n): 
                _bool(out2D[i], p, shift + 97*<UINT32>i)
            _bool(out2D[:,0], n, x)
    else:
        out = np.zeros(size, dtype = np.uint8)
        _bool(out, size, x)
    return np.asarray(out2D).view(np.bool) if isTuple else np.asarray(out).view(np.bool)


######
cdef ARRAY uint8_(int left, int right, size, int seed, LONG div, int diff):
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
                _uint8(out2D[i], p, change + 97*<UINT32>i, mult, left, div, diff)
    else:
        out = np.zeros(size, dtype = np.uint8)
        _uint8(out, size, x, mult, left, div, diff)

    return np.asarray(out2D) if isTuple else np.asarray(out)


######
cdef ARRAY uint16_(int left, int right, size, int seed, LONG div, int diff):
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
                _uint16(out2D[i], p, change + 97*<UINT32>i, mult, left, div, diff)
    else:
        out = np.zeros(size, dtype = np.uint16)
        _uint16(out, size, x, mult, left, div, diff)

    return np.asarray(out2D) if isTuple else np.asarray(out)


######
cdef ARRAY uint32_(int left, int right, size, int seed, LONG div, int diff):
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
                _uint32(out2D[i], p, change + 97*<UINT32>i, mult, left, div, diff)
    else:
        out = np.zeros(size, dtype = np.uint32)
        _uint32(out, size, x, mult, left, div, diff)

    return np.asarray(out2D) if isTuple else np.asarray(out)


######
cdef ARRAY uint64_(int left, int right, size, int seed, LONG div, int diff):
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
                _uint64(out2D[i], p, change + 97*<UINT64>i, mult, left, div, diff)
    else:
        out = np.zeros(size, dtype = np.uint64)
        _uint64(out, size, x, mult, left, div, diff)

    return np.asarray(out2D) if isTuple else np.asarray(out)


######
cdef ARRAY int8_(int left, int right, size, int seed, LONG div, int diff):
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
                _int8(out2D[i], p, change + 97*<UINT32>i, mult, shift, div, diff)
    else:
        out = np.zeros(size, dtype = np.int8)
        _int8(out, size, x, mult, shift, div, diff)

    return np.asarray(out2D) if isTuple else np.asarray(out)


######
cdef ARRAY int16_(int left, int right, size, int seed, LONG div, int diff):
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
                _int16(out2D[i], p, change + 97*<UINT32>i, mult, shift, div, diff)
    else:
        out = np.zeros(size, dtype = np.int16)
        _int16(out, size, x, mult, shift, div, diff)

    return np.asarray(out2D) if isTuple else np.asarray(out)


######
cdef ARRAY int32_(int left, int right, size, int seed, LONG div, int diff):
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
                _int32(out2D[i], p, change + 97*<UINT32>i, mult, shift, div, diff)
    else:
        out = np.zeros(size, dtype = np.int32)
        _int32(out, size, x, mult, shift, div, diff)

    return np.asarray(out2D) if isTuple else np.asarray(out)


######
cdef ARRAY int64_(int left, int right, size, int seed, LONG div, int diff):
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
                _int64(out2D[i], p, change + 97*<UINT64>i, mult, shift, div, diff)
    else:
        out = np.zeros(size, dtype = np.int64)
        _int64(out, size, x, mult, shift, div, diff)

    return np.asarray(out2D) if isTuple else np.asarray(out)


######
cdef ARRAY float32_(int left, int right, size, int seed, LONG div, int diff):
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
                _float32(out2D[i], p, change + 97*<UINT32>i, mult, shift, div, diff)
    else:
        out = np.zeros(size, dtype = np.float32)
        _float32(out, size, x, mult, shift, div, diff)

    return np.asarray(out2D) if isTuple else np.asarray(out)


######
cdef ARRAY float64_(int left, int right, size, int seed, LONG div, int diff):
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
                _float64(out2D[i], p, change + 97*<UINT64>i, mult, shift, div, diff)
    else:
        out = np.zeros(size, dtype = np.float64)
        _float64(out, size, x, mult, shift, div, diff)

    return np.asarray(out2D) if isTuple else np.asarray(out)


######
cpdef ARRAY uniform(
    int left = 0, int right = 10, size = 10, int seed = -1, dtype = np.float32):
    
    cdef char dt = ord(np.dtype(dtype).char)
    cdef LONG div, n, p
    cdef int diff

    assert right > left

    if type(size) is tuple:
        n, p = <LONG> size[0], <LONG> size[1]
        assert n > 0 and p > 0
        div = p // 4
        diff = p-4*div
    else:
        size = <LONG>size
        assert size > 0
        div = size // 4
        diff = size-4*div

    cdef int l, r
    l, r = <int> fabs(left), <int> fabs(right)
    cdef int _max = l if l > r else r

    if dt == boolean:
        return bool_(size, seed)

    ###
    if dt == uint8:
        if left < 0: left = 0
        if right > uint8_max: dt = uint16
        else: return uint8_(left, right, size, seed, div, diff)

    if dt == uint16:
        if left < 0: left = 0
        if right > uint16_max: dt = uint32
        else: return uint16_(left, right, size, seed, div, diff)

    if dt == uint32 or dt == cuint:
        if left < 0: left = 0
        if right > uint32_max: dt = uint64
        else: return uint32_(left, right, size, seed, div, diff)

    if dt == uint64 or dt == upointer:
        if left < 0: left = 0
        else: return uint64_(left, right, size, seed, div, diff)

    ###
    if dt == int8:
        if right > int8_max: dt = int16
        else: return int8_(left, right, size, seed, div, diff)

    if dt == int16:
        if right > int16_max: dt = int32
        else: return int16_(left, right, size, seed, div, diff)

    if dt == int32 or dt == cint:
        if right > int32_max: dt = int64
        else: return int32_(left, right, size, seed, div, diff)

    if dt == int64 or dt == pointer:
        return int64_(left, right, size, seed, div, diff)

    ###
    if dt == float32:
        if right > float32_max: dt = float64
        else: return float32_(left, right, size, seed, div, diff)

    return float64_(left, right, size, seed, div, diff)
