# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import numpy as np
from numba import autojit, jit, double, void, uint32, npy_intp, typeof

def uint_int_div_ary(elts, normdist, seed):
    for i in xrange(elts.shape[0]):
        # Problem with using sext instead of zext for uint32
        elt = (seed[i] // uint32(normdist.shape[0]))
        elts[i] = elt

def test_uint_int_div_ary():
    NPATHS = 10
    normdist = np.empty(1000) #np.random.normal(0., 1., 1000)
    seed = np.arange(0x80000000, 0x80000000 + NPATHS, dtype=np.uint32)

    gold = np.empty(NPATHS, dtype=np.int32)
    got = gold.copy()
    uint_int_div_ary(gold, normdist, seed)

    print('expect %s' % gold)
    sig = void(uint32[:], double[:], uint32[:])
    numba_func = jit(sig)(uint_int_div_ary)
    numba_func(got, normdist, seed)
    print('got %s' % got)

    assert all(gold == got)

if __name__ == '__main__':
    test_uint_int_div_ary()
