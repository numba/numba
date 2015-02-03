from __future__ import print_function

import random

import numpy as np

import numba.unittest_support as unittest
from numba import jit, _helperlib
from .support import TestCase


# State size of the Mersenne Twister
N = 624

py_state_ptr = _helperlib.c_helpers['py_random_state']
np_state_ptr = _helperlib.c_helpers['np_random_state']


@jit(nopython=True)
def random_seed(x):
    random.seed(x)
    return 0

@jit(nopython=True)
def random_random():
    return random.random()

@jit(nopython=True)
def random_getrandbits(b):
    return random.getrandbits(b)

@jit(nopython=True)
def random_gauss(mu, sigma):
    return random.gauss(mu, sigma)


def _copy_py_state(r, ptr):
    """
    Copy state of Python state *r* to Numba state *ptr*.
    """
    mt = r.getstate()[1]
    ints, index = mt[:-1], mt[-1]
    _helperlib.rnd_set_state(ptr, (index, list(ints)))
    return ints, index

def _copy_np_state(r, ptr):
    """
    Copy state of Numpy state *r* to Numba state *ptr*.
    """
    ints, index = r.get_state()[1:3]
    _helperlib.rnd_set_state(ptr, (index, [int(x) for x in ints]))
    return ints, index

def sync_to_numpy(r):
    _ver, mt_st, _gauss_next = r.getstate()
    mt_pos = mt_st[-1]
    mt_ints = mt_st[:-1]
    assert len(mt_ints) == 624

    np_st = ('MT19937', np.array(mt_ints, dtype='uint32'), mt_pos)
    if _gauss_next is None:
        np_st += (0, 0.0)
    else:
        np_st += (1, _gauss_next)

    np.random.set_state(np_st)


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
        ints, index = _copy_py_state(r, ptr)
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
            ints = list(st[1])
            index = st[2]
            assert index == N  # sanity check
            _helperlib.rnd_init(ptr, i)
            self.assertEqual(_helperlib.rnd_get_state(ptr), (index, ints))

    def test_get_set_state(self):
        self.check_get_set_state(py_state_ptr)

    def test_shuffle(self):
        self.check_shuffle(py_state_ptr)

    def test_init(self):
        self.check_init(py_state_ptr)


class TestRandom(TestCase):

    def check_random(self, func):
        """
        Check a random()-like function.
        """
        r = np.random.RandomState()
        for i in [0, 1, 125, 2**32 - 1]:
            r.seed(i)
            random_seed(i)
            # Be sure to trigger a reshuffle
            for j in range(N + 10):
                self.assertPreciseEqual(func(), r.uniform(0.0, 1.0))

    def test_random_random(self):
        self.check_random(random_random)

    def check_getrandbits(self, func, ptr):
        """
        Check a getrandbits()-like function.
        """
        # Our implementation follows CPython's for bits <= 64.
        r = random.Random()
        _copy_py_state(r, ptr)
        for nbits in range(1, 65):
            expected = r.getrandbits(nbits)
            got = func(nbits)
            self.assertPreciseEqual(expected, got)

    def test_random_getrandbits(self):
        self.check_getrandbits(random_getrandbits, py_state_ptr)

    def check_gauss(self, func, ptr):
        """
        Check a gauss()-like function.
        """
        # Our implementation follows Numpy's.
        r = np.random.RandomState()
        _copy_np_state(r, ptr)
        for mu, sigma in [(1.0, 1.0), (2.0, 0.5), (-2.0, 0.5)]:
            for i in range(N // 2 + 10):
                self.assertPreciseEqual(func(mu, sigma), r.normal(mu, sigma))

    def test_random_gauss(self):
        self.check_gauss(random_gauss, py_state_ptr)


if __name__ == "__main__":
    unittest.main()

