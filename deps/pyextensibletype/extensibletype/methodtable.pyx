from libc cimport stdlib
cimport numpy as cnp
import numpy as np

from extensibletype cimport *

import intern

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


cdef class Hasher(object):
    """
    Generate a globally unique hashes for signature strings.
    """

    def hash_signature(self, signature):
        cdef uint64_t hashvalue
        # cdef bytes md5 = hashlib.md5(signature).digest()
        # (&hashvalue)[0] = (<uint64_t *> <char *> md5)[0]
        hashvalue = intern.global_intern(signature)

        return hashvalue


cdef class PerfectHashMethodTable(object):
    """
    Simple wrapper for hash-based virtual method tables.
    """

    cdef PyCustomSlots_Table *table
    cdef uint16_t *displacements
    cdef Hasher hasher

    def __init__(self, hasher):
        self.hasher = hasher

    def generate_table(self, n, ids, flags, funcs):
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

            hashes[i] = self.hasher.hash_signature(signature)

        # Perfect hash our table
        PyCustomSlots_PerfectHash(self.table, &hashes[0])

    def find_method(self, signature):
        """
        Find method of the given signature. Use from non-performance
        critical code.
        """
        cdef uintptr_t id = intern.global_intern(signature)
        cdef uint64_t prehash = self.hasher.hash_signature(signature)

        cdef int idx = (((prehash >> self.table.r) & self.table.m_f) ^
                        self.displacements[prehash & self.table.m_g])

        assert 0 <= idx < self.size

        if <uintptr_t> self.table.entries[idx].id != id:
            return None

        return (<uintptr_t> self.table.entries[idx].ptr,
                self.table.entries[idx].flags)

    def __dealloc__(self):
        stdlib.free(self.table)
        self.table = NULL

    property table_ptr:
        def __get__(self):
            return <uintptr_t> self.table

    property size:
        def __get__(self):
            return self.table.n

