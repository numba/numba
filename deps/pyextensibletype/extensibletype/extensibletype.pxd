cdef extern from "pstdint.h":
    ctypedef unsigned int uint32_t
    ctypedef unsigned long long uint64_t
    ctypedef unsigned short uint16_t
    ctypedef unsigned char uint8_t
    ctypedef uint64_t uintptr_t

cdef extern from "perfecthash.h":
    ctypedef struct PyCustomSlots_Entry:
        uint64_t id
        void *ptr

    ctypedef struct PyCustomSlots_Table:
        uint64_t flags
        uint64_t m_f, m_g
        PyCustomSlots_Entry *entries
        uint16_t n, b
        uint8_t r

    ctypedef struct PyCustomSlots_Table_64_64:
        PyCustomSlots_Table base
        uint16_t d[64]
        PyCustomSlots_Entry entries_mem[64]


    int PyCustomSlots_PerfectHash(PyCustomSlots_Table *table, uint64_t *hashes)

