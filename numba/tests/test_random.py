from __future__ import print_function

import random

import numpy as np

import numba.unittest_support as unittest
from numba import _helperlib
from .support import TestCase


N = 624

py_state_ptr = _helperlib.c_helpers['py_random_state']
np_state_ptr = _helperlib.c_helpers['np_random_state']


class TestInternals(TestCase):
    """
    Test low-level internals of the implementation.
    """

    def check_get_set_state(self, ptr):
        state = _helperlib.rnd_get_state(ptr)
        i, ints = state
        self.assertIsInstance(i, int)
        self.assertIsInstance(ints, list)
        self.assertEqual(len(ints), N)
        j = (i * 100007) % N
        ints = [i * 3 for i in range(N)]
        # Roundtrip
        _helperlib.rnd_set_state(ptr, (j, ints))
        self.assertEqual(_helperlib.rnd_get_state(ptr), (j, ints))

    def check_shuffle(self, ptr):
        # We test shuffling against CPython
        r = random.Random()
        mt = r.getstate()[1]
        ints, index = mt[:-1], mt[-1]
        _helperlib.rnd_set_state(ptr, (index, list(ints)))
        # Force shuffling in CPython generator
        for i in range(index, N + 1, 2):
            r.random()
        _helperlib.rnd_shuffle(ptr)
        # Check new integer keys
        mt = r.getstate()[1]
        ints, index = mt[:-1], mt[-1]
        self.assertEqual(_helperlib.rnd_get_state(ptr)[1], list(ints))

    def check_init(self, ptr):
        # We use the same integer seeding as Numpy
        # (CPython is different: it treats the integer as a byte array)
        r = np.random.RandomState()
        for i in [0, 1, 125, 2**32 - 5]:
            r.seed(i)
            st = r.get_state()
            ints = list(r[1])
            index = r[2]
            assert index == N  # sanity check
            _helperlib.rnd_init(ptr, i)
            self.assertEqual(_helperlib.rnd_get_state(ptr), (index, ints))

    def test_get_set_state(self):
        self.check_get_set_state(py_state_ptr)

    def test_shuffle(self):
        self.check_shuffle(py_state_ptr)

    def test_init(self):
        self.check_shuffle(py_state_ptr)


if __name__ == "__main__":
    unittest.main()

