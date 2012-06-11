cimport numpy as cnp

cdef extern from "stdint.h":
    ctypedef unsigned long long uint64_t
    ctypedef unsigned short uint16_t
    ctypedef unsigned char uint8_t
    ctypedef uint64_t uintptr_t

cdef extern from "perfecthash.h":
    ctypedef struct PyCustomSlots_Entry:
        char *id
        uintptr_t flags
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
    void _PyCustomSlots_bucket_argsort(uint16_t *p, uint8_t *binsizes,
                                       uint8_t *number_of_bins_by_size)

def bucket_argsort(cnp.ndarray[uint16_t, mode='c'] p,
                   cnp.ndarray[uint8_t, mode='c'] binsizes,
                   cnp.ndarray[uint8_t, mode='c'] number_of_bins_by_size):
    _PyCustomSlots_bucket_argsort(&p[0], &binsizes[0],
                                  &number_of_bins_by_size[0])

def test(cnp.ndarray[uint64_t] hashes):
    cdef PyCustomSlots_Table_64_64 table
    table.base.flags = 0
    table.base.n = 64
    table.base.b = 64
    table.base.entries = &table.entries_mem[0]
    for i in range(64):
        table.entries_mem[i].id = NULL
        table.entries_mem[i].flags = i
        table.entries_mem[i].ptr = NULL

    PyCustomSlots_PerfectHash(&table.base, &hashes[0])

    for i in range(64):
        print table.entries_mem[i].flags, table.d[i]
