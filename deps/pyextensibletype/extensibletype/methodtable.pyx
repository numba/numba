from libc cimport stdlib
cimport numpy as cnp
import numpy as np

from extensibletype cimport *
from . import extensibletype

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

class HashingError(Exception):
    """
    Raised when we can't create a perfect hash-based function table.
    """

cdef PyCustomSlots_Table *allocate_hash_table(uint16_t size) except NULL:
    cdef PyCustomSlots_Table *table
    cdef uint16_t nbins

    size = roundup(size)
    assert size * 4 <= 0xFFFF, hex(size)
    nbins = size * 4

    table = <PyCustomSlots_Table *> stdlib.calloc(
        1, sizeof(PyCustomSlots_Table) + sizeof(uint16_t) * nbins +
           sizeof(PyCustomSlots_Entry) * size)

    if table == NULL:
        raise MemoryError

    table.n = size
    table.b = nbins
    table.flags = 0

    assert table.b >= table.n, (table.b, table.n, nbins)

    table.entries = <PyCustomSlots_Entry *> (
        (<char *> table) + sizeof(PyCustomSlots_Table) +
        nbins * sizeof(uint16_t))

    return table

def make_bytes(s):
    if isinstance(s, str):
        # Python 3
        s = s.encode("ascii")

    return s

cdef class Hasher(object):
    """
    Generate a globally unique hashes for signature strings.
    """

    def hash_signature(self, signature):
        cdef uint64_t hashvalue
        # cdef bytes md5 = hashlib.md5(signature).digest()
        # (&hashvalue)[0] = (<uint64_t *> <char *> md5)[0]

        hashvalue = intern.global_intern(make_bytes(signature))
        return hashvalue


cdef class PerfectHashMethodTable(object):
    """
    Simple wrapper for hash-based virtual method tables.
    """

    cdef PyCustomSlots_Table *table
    cdef uint16_t *displacements
    cdef Hasher hasher

    cdef object id_to_signature, signatures

    def __init__(self, hasher):
        self.hasher = hasher
        # For debugging
        self.id_to_signature = {}

    def generate_table(self, n, ids, flags, funcs, method_names=None):
        cdef Py_ssize_t i
        cdef cnp.ndarray[uint64_t] hashes

        self.table = allocate_hash_table(n)
        self.displacements = <uint16_t *> (<char *> self.table +
                                               sizeof(PyCustomSlots_Table))

        hashes = np.zeros(self.table.n, dtype=np.uint64)

        intern.global_intern_initialize()

        # Initialize hash table entries, build hash ids
        assert len(ids) == len(flags) == len(funcs)

        for i, (signature, flag, func) in enumerate(zip(ids, flags, funcs)):
            id = self.hasher.hash_signature(signature)

            self.table.entries[i].id = id
            self.table.entries[i].ptr = <void *> <uintptr_t> func

            hashes[i] = id
            self.id_to_signature[id] = signature


        hashes[n:self.table.n] = extensibletype.draw_hashes(np.random,
                                                            self.table.n - n)
        # print "n", n, "table.n", self.table.n, "table.b", self.table.b
        assert len(np.unique(hashes)) == len(hashes)

        # print "-----------------------"
        # print self
        # print "-----------------------"

        assert self.table.b >= self.table.n, (self.table.b, self.table.n)

        # Perfect hash our table
        if PyCustomSlots_PerfectHash(self.table, &hashes[0]) < 0:
            # TODO: sensible error messages
            raise HashingError(
                "Unable to create perfect hash table for table: %s" % self)

        for i, signature in enumerate(ids):
            assert self.find_method(signature) is not None, (i, signature)

        # For debugging
        self.signatures = ids

    def find_method(self, signature):
        """
        Find method of the given signature. Use from non-performance
        critical code.
        """
        cdef uint64_t prehash = intern.global_intern(make_bytes(signature))

        assert 0 <= self.displacements[prehash & self.table.m_g] < self.table.b
        cdef uint64_t idx = (((prehash >> self.table.r) & self.table.m_f) ^
                             self.displacements[prehash & self.table.m_g])

        assert 0 <= idx < self.size, (idx, self.size)

        if self.table.entries[idx].id != prehash:
            return None

        return (<uintptr_t> self.table.entries[idx].ptr,
                self.table.entries[idx].id & 0xFF)

    def __str__(self):
        buf = ["PerfectHashMethodTable("]
        for i in range(self.table.n):
            id = self.table.entries[i].id
            ptr = <uintptr_t> self.table.entries[i].ptr
            sig = self.id_to_signature.get(id, "<empty>")
            s = "    id: 0x%-16x  funcptr: %20d  signature: %s" % (id, ptr, sig)
            buf.append(s)

        buf.append(")")

        return "\n".join(buf)

    def __dealloc__(self):
        # stdlib.free(self.table)
        # self.table = NULL
        pass

    property table_ptr:
        def __get__(self):
            return <uintptr_t> self.table

    property size:
        def __get__(self):
            return self.table.n

