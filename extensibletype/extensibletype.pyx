cimport stdlib
cimport numpy as cnp
import numpy as np

import hashlib

from . import intern

cdef extern from "perfecthash.h":
    void _PyCustomSlots_bucket_argsort(uint16_t *p, uint8_t *binsizes,
                                       uint8_t *number_of_bins_by_size)

def bucket_argsort(cnp.ndarray[uint16_t, mode='c'] p,
                   cnp.ndarray[uint8_t, mode='c'] binsizes,
                   cnp.ndarray[uint8_t, mode='c'] number_of_bins_by_size):
    _PyCustomSlots_bucket_argsort(&p[0], &binsizes[0],
                                  &number_of_bins_by_size[0])

def perfect_hash(cnp.ndarray[uint64_t] hashes, int repeat=1):
    """Used for testing. Takes the hashes as input, and returns
       a permutation array and hash parameters:

       (p, r, m_f, m_g, d)
    """
    cdef PyCustomSlots_Table_64_64 table
    table.base.flags = 0
    table.base.n = 64
    table.base.b = 64
    table.base.entries = &table.entries_mem[0]
    for i in range(64):
        table.entries_mem[i].id = NULL
        table.entries_mem[i].flags = i
        table.entries_mem[i].ptr = NULL

    cdef int r
    for r in range(repeat):
        PyCustomSlots_PerfectHash(&table.base, &hashes[0])

    d = np.zeros(64, dtype=np.uint16)
    p = np.zeros(64, dtype=np.uint16)

    for i in range(64):
        p[i] = table.entries_mem[i].flags
        d[i] = table.d[i]

    return p, table.base.r, table.base.m_f, table.base.m_g, d

def roundup(x):
    "Round up to a power of two"
    x -= 1
    x |= x >> 1
    x |= x >> 2
    x |= x >> 4
    x |= x >> 8
    x |= x >> 16
    x |= x >> 32
    x += 1
    return x

cdef PyCustomSlots_Table *allocate_hash_table(uint16_t size) except NULL:
    cdef PyCustomSlots_Table *table

    size = roundup(size)

    table = <PyCustomSlots_Table *> stdlib.malloc(
        sizeof(PyCustomSlots_Table) + sizeof(uint16_t) * size +
        sizeof(PyCustomSlots_Entry) * size)

    if table == NULL:
        raise MemoryError

    table.n = size
    table.b = size
    table.flags = 0

    table.entries = <PyCustomSlots_Entry *> ((<char *> &table[1]) +
                                             size * sizeof(uint16_t))

    return table

cdef class PerfectHashMethodTable(object):
    """
    Simple wrapper for hash-based virtual method tables.
    """

    cdef PyCustomSlots_Table *table
    cdef uint16_t *displacements

    def __init__(self, n, ids, flags, funcs):
        cdef Py_ssize_t i
        cdef cnp.ndarray[uint64_t] hashes

        self.table = allocate_hash_table(n)
        self.displacements = <uint16_t *> (<char *> self.table +
                                               sizeof(PyCustomSlots_Table))

        hashes = np.empty(n, dtype=np.uint64)

        intern.global_intern_initialize()

        # Initialize hash table entries, build hash ids
        for i, (signature, flag, func) in enumerate(zip(ids, flags, funcs)):
            id = intern.global_intern(signature)

            self.table.entries[i].id = <char *> <uintptr_t> id
            self.table.entries[i].flags = flag
            self.table.entries[i].ptr = <void *> <uintptr_t> func

            hashes[i] = self.hash(signature)

        # Perfect hash our table
        PyCustomSlots_PerfectHash(self.table, &hashes[0])

    def hash(self, signature):
        cdef uint64_t hashvalue
        cdef bytes md5 = hashlib.md5(signature).digest()

        (&hashvalue)[0] = (<uint64_t *> <char *> md5)[0]
        return hashvalue

    def find_method(self, signature):
        """
        Find method of the given signature. Use from non-performance
        critical code.
        """
        cdef uintptr_t id = intern.global_intern(signature)
        cdef uint64_t prehash = self.hash(signature)

        cdef int idx = (((prehash >> self.table.r) & self.table.m_f) ^
                        self.displacements[prehash & self.table.m_g])

        assert 0 <= idx < self.size

        if <uintptr_t> self.table.entries[idx].id != id:
            return None

        return (<uintptr_t> self.table.entries[idx].ptr,
                self.table.entries[idx].flags)

    property table_ptr:
        def __get__(self):
            return <uintptr_t> self.table

    property size:
        def __get__(self):
            return self.table.n


#cdef extern from "md5sum.h":
#    ctypedef struct MD5_CTX:
#        uint32_t i[2]
#        uint32_t buf[4]
#        unsigned char in_ "in"[64]
#        unsigned char digest[16]
#
#    void MD5Init(MD5_CTX *mdContext)
#    void MD5Update(MD5_CTX *mdContext, unsigned char *inBuf,
#                   unsigned int inLen)
#    void MD5Final(MD5_CTX *mdContext)
#
#cdef extern from "hash.h":
#    uint64_t hash_crapwow64(unsigned char *buf, uint64_t len, uint64_t seed)
#
#def crapwowbench(int repeat=1):
#    cdef int r
#    cdef MD5_CTX ctx
#    for r in range(repeat):
#        hash_crapwow64("asdf", 4, 0xf123456781234567)
#
#
#def md5bench(int repeat=1):
#    cdef int r
#    cdef MD5_CTX ctx
#    for r in range(repeat):
#        MD5Init(&ctx)
#        MD5Update(&ctx, "asdf", 4)
#        MD5Final(&ctx)

