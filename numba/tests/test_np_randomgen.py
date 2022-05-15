import numba
import numpy as np

from numba import types
from numba.tests.support import TestCase
from numba.np.random.generator_methods import get_proper_func
from numpy.random import MT19937, Generator
from numpy.testing import assert_equal, assert_array_equal, assert_raises


def test_proper_func_provider():
    def test_32bit_func():
        return 32

    def test_64bit_func():
        return 64

    assert_equal(get_proper_func(test_32bit_func, test_64bit_func,
                                 np.float64)[0](), 64)
    assert_equal(get_proper_func(test_32bit_func, test_64bit_func,
                                 np.float32)[0](), 32)
    assert_equal(get_proper_func(test_32bit_func, test_64bit_func,
                                 types.float64)[0](), 64)
    assert_equal(get_proper_func(test_32bit_func, test_64bit_func,
                                 types.float32)[0](), 32)

    # With any other datatype it should return a TypeError
    with assert_raises(TypeError):
        get_proper_func(test_32bit_func, test_64bit_func, np.int32)


class TestRandomGenerators(TestCase):
    def check_numpy_parity(self, distribution_func,
                           bitgen_instance=None, seed=1,
                           test_sizes=None, test_dtypes=None):

        distribution_func = numba.njit(distribution_func)
        if bitgen_instance is None:
            numba_rng_instance = np.random.default_rng(seed=seed)
            numpy_rng_instance = np.random.default_rng(seed=seed)
        else:
            numba_rng_instance = Generator(bitgen_instance(seed))
            numpy_rng_instance = Generator(bitgen_instance(seed))

        if test_sizes is None:
            test_sizes = [None, (), (100,), (10,20,30)]
        if test_dtypes is None:
            test_dtypes = [np.float32, np.float64]

        # Check parity for different size cases
        for size in test_sizes:
            for dtype in test_dtypes:
                numba_res = distribution_func(numba_rng_instance,
                                              size, dtype)
                numpy_res = distribution_func.py_func(numpy_rng_instance,
                                                      size, dtype)

                assert_array_equal(numba_res, numpy_res)

        # Check if the end state of both BitGenerators is same
        # after drawing the distributions
        numba_gen_state = numba_rng_instance.__getstate__()['state']
        numpy_gen_state = numpy_rng_instance.__getstate__()['state']

        for _state_key in numpy_gen_state:
            assert_equal(numba_gen_state[_state_key],
                         numpy_gen_state[_state_key])

    def test_random(self):
        # Test with no arguments
        dist_func = lambda x, size, dtype:x.random()
        # Provide single values so this test would run exactly once
        self.check_numpy_parity(dist_func, test_sizes=[100],
                                test_dtypes=[np.float64])
        self.check_numpy_parity(dist_func, bitgen_instance=MT19937)

        dist_func = lambda x, size, dtype:x.random(size=size, dtype=dtype)
        self.check_numpy_parity(dist_func)
        self.check_numpy_parity(dist_func, bitgen_instance=MT19937)
