cdef extern from "pstdint.h":
    ctypedef int int8_t
    ctypedef int int16_t
    ctypedef int int32_t
    ctypedef int int64_t

    ctypedef int uint8_t
    ctypedef int uint16_t
    ctypedef int uint32_t
    ctypedef int uint64_t

    ctypedef int intptr_t
    ctypedef int uintptr_t

def test_pstdint():
    assert sizeof(int8_t) == sizeof(uint8_t) == 1
    assert sizeof(int16_t) == sizeof(uint16_t) == 2
    assert sizeof(int32_t) == sizeof(uint32_t) == 4
    assert sizeof(int64_t) == sizeof(uint64_t) == 8

    assert sizeof(intptr_t) == sizeof(uintptr_t) >= sizeof(void *)
