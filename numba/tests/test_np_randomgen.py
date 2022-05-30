import numba
import numpy as np

from numba import types
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.np.random.generator_methods import _get_proper_func
from numba.np.random.generator_core import next_uint32, next_uint64, next_double
from numpy.random import MT19937, Generator
from numba.core.errors import TypingError


class TestHelperFuncs(TestCase):
    def test_proper_func_provider(self):
        def test_32bit_func():
            return 32

        def test_64bit_func():
            return 64

        self.assertEqual(_get_proper_func(test_32bit_func, test_64bit_func,
                         np.float64)[0](), 64)
        self.assertEqual(_get_proper_func(test_32bit_func, test_64bit_func,
                         np.float32)[0](), 32)
        self.assertEqual(_get_proper_func(test_32bit_func, test_64bit_func,
                         types.float64)[0](), 64)
        self.assertEqual(_get_proper_func(test_32bit_func, test_64bit_func,
                         types.float32)[0](), 32)

        # With any other datatype it should return a TypeError
        with self.assertRaises(TypingError) as raises:
            _get_proper_func(test_32bit_func, test_64bit_func, np.int32)
        self.assertIn(
            'Unsupported dtype int32 for the given distribution',
            str(raises.exception)
        )


class TestRandomGenerators(MemoryLeakMixin, TestCase):
    def check_numpy_parity(self, distribution_func,
                           bitgen_type=None, seed=1,
                           test_sizes=None, test_dtypes=None):

        distribution_func = numba.njit(distribution_func)
        if bitgen_type is None:
            numba_rng_instance = np.random.default_rng(seed=seed)
            numpy_rng_instance = np.random.default_rng(seed=seed)
        else:
            numba_rng_instance = Generator(bitgen_type(seed))
            numpy_rng_instance = Generator(bitgen_type(seed))

        if test_sizes is None:
            test_sizes = [None, (), (100,), (10, 20, 30)]
        if test_dtypes is None:
            test_dtypes = [np.float32, np.float64]

        # Check parity for different size cases
        for size in test_sizes:
            for dtype in test_dtypes:
                numba_res = distribution_func(numba_rng_instance,
                                              size, dtype)
                numpy_res = distribution_func.py_func(numpy_rng_instance,
                                                      size, dtype)

                self.assertPreciseEqual(numba_res, numpy_res)

        # Check if the end state of both BitGenerators is same
        # after drawing the distributions
        numba_gen_state = numba_rng_instance.__getstate__()['state']
        numpy_gen_state = numpy_rng_instance.__getstate__()['state']

        for _state_key in numpy_gen_state:
            self.assertPreciseEqual(numba_gen_state[_state_key],
                                    numpy_gen_state[_state_key])

    def _test_bitgen_func_parity(self, func_name, bitgen_func, seed=1):
        numba_rng_instance = np.random.default_rng(seed=seed)
        numpy_rng_instance = np.random.default_rng(seed=seed)

        numpy_func = getattr(numpy_rng_instance.bit_generator.ctypes, func_name)
        numpy_res = numpy_func(numpy_rng_instance.bit_generator.ctypes.state)

        numba_func = numba.njit(lambda x: bitgen_func(x.bit_generator))
        numba_res = numba_func(numba_rng_instance)

        self.assertPreciseEqual(numba_res, numpy_res)

    def test_npgen_boxing_unboxing(self):
        rng_instance = np.random.default_rng()
        numba_func = numba.njit(lambda x: x)
        self.assertEqual(rng_instance, numba_func(rng_instance))
        self.assertEqual(id(rng_instance), id(numba_func(rng_instance)))

    def test_bitgen_funcs(self):
        self._test_bitgen_func_parity("next_uint32", next_uint32)
        self._test_bitgen_func_parity("next_uint64", next_uint64)
        self._test_bitgen_func_parity("next_double", next_double)

    def test_random(self):
        # Test with no arguments
        dist_func = lambda x, size, dtype:x.random()
        # Provide single values so this test would run exactly once
        self.check_numpy_parity(dist_func, test_sizes=[100],
                                test_dtypes=[np.float64])
        self.check_numpy_parity(dist_func, bitgen_type=MT19937)

        dist_func = lambda x, size, dtype:x.random(size=size, dtype=dtype)
        self.check_numpy_parity(dist_func)
        self.check_numpy_parity(dist_func, bitgen_type=MT19937)
