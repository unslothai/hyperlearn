
cdef extern from "immintrin.h":
    ctypedef char __m128i
    ctypedef float __m128
    
    __m128  _mm_load_ps1 (float* mem_addr) nogil
    void    _mm_store_ps1 (float* mem_addr, __m128 a) nogil
    
    __m128i _mm_and_si128 (__m128i __A, __m128i __B) nogil
    __m128  _mm_fmadd_ps (__m128 a, __m128 b, __m128 c) nogil

ctypedef __m128 SSEf

cdef (SSEf, SSEf) _store(float mult, float shift) nogil:
    cdef SSEf _mult, _shift
    return _mm_load_ps1(&mult), _mm_load_ps1(&shift)

cdef void _mult_add(SSEf mult, float* x, SSEf shift) nogil:
    _mm_store_ps1(x, _mm_fmadd_ps(mult, _mm_load_ps1(x), shift))
    

cdef float mult = 10.0/(0x7FFF + 1)
cdef float shift = <float> 10
cdef SSEf _mult, _shift
_mult, _shift = _store(mult, shift)

cpdef fast():
    cdef int i
    cdef float x = 30000
    cdef float* _x = &x
    
    with nogil:
        for i in range(100000000):
            _mult_add(_mult, _x, _shift)


cpdef slow():
    cdef int i
    cdef float x = 30000
    
    with nogil:
        for i in range(100000000):
            x = (x*mult + shift)

