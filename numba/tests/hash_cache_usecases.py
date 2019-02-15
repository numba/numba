# -*- coding: utf-8 -*-
"""
This file will be copied to a temporary directory in order to
exercise caching compiled Numba functions.

See test_hashing.py's TestHashInCache()
"""
from __future__ import division, print_function, absolute_import
import sys

import numpy as np
from numba import jit, utils
from numba.tests.support import TestCase


@jit(cache=True, nopython=True)
def simple_usecase(x):
    return hash(x)


class _TestModule(TestCase):
    """
    Tests for functionality of this module's function.
    Note this does not define any "test_*" method, instead check_module()
    should be called by hand.
    """

    def check_module(self, mod, assert_run_from_cache=False):
        f = mod.simple_usecase
        ints = [np.uint8(123), np.int16(123), np.uint32(123), np.uint64(123)]
        floats = [np.float32(123), np.float64(123), np.complex64(123 + 456j),
                  np.complex128(123 + 456j)]
        tuples = [(1, 2, 3), (1.2, 3j, 4)]

        inputs = ints + floats + tuples

        if utils.IS_PY3:
            strings = ['numba', "眼" , "🐍⚡"]
            inputs.extend(strings)

        for i in inputs:
            self.assertPreciseEqual(simple_usecase(i), hash(i))

        if assert_run_from_cache:
            ntypes = 10
            ndata = 1
            expected = ntypes + ndata
            self.assertEqual(sum(f.stats.cache_hits.values()), expected)

    # For 2.x
    def runTest(self):
        raise NotImplementedError


def self_test(**kwargs):
    mod = sys.modules[__name__]
    _TestModule().check_module(mod, **kwargs)
