cimport numpy as cnp
import numpy as np

import hashlib

cdef extern from "perfecthash.h":
    void _PyCustomSlots_bucket_argsort(uint16_t *p, uint8_t *binsizes,
                                       uint16_t *number_of_bins_by_size)

def bucket_argsort(cnp.ndarray[uint16_t, mode='c'] p,
                   cnp.ndarray[uint8_t, mode='c'] binsizes,
                   cnp.ndarray[uint16_t, mode='c'] number_of_bins_by_size):
    _PyCustomSlots_bucket_argsort(&p[0], &binsizes[0],
                                  &number_of_bins_by_size[0])

def get_random_hashes(rng, nitems):
    return rng.randint(-2**31, 2**31-1, size=nitems).astype(np.uint64)

def draw_hashes(rng, nitems):
    assert sizeof(long) >= 4

    hashes = get_random_hashes(rng, nitems)
    hashes <<= 32
    hashes |= get_random_hashes(rng, nitems)

    return hashes

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
        table.entries_mem[i].id = hashes[i]
        table.entries_mem[i].ptr = NULL

    cdef int r
    for r in range(repeat):
        PyCustomSlots_PerfectHash(&table.base, &hashes[0])

    d = np.zeros(64, dtype=np.uint16)
    p = np.zeros(64, dtype=np.uint16)

    for i in range(64):
        p[i] = table.entries_mem[i].id & 0xFF
        d[i] = table.d[i]

    return p, table.base.r, table.base.m_f, table.base.m_g, d

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

