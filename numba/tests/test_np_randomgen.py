import numba
import numpy as np
import sys

from numba import types
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.np.random.generator_methods import _get_proper_func
from numba.np.random.generator_core import next_uint32, next_uint64, next_double
from numpy.random import MT19937, Generator
from numba.core.errors import TypingError
from numba.tests.support import run_in_new_process_caching, SerialMixin


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


def test_generator_caching():
    nb_rng = np.random.default_rng(1)
    np_rng = np.random.default_rng(1)
    py_func = lambda x: x.random(10)
    numba_func = numba.njit(cache=True)(py_func)
    assert np.allclose(np_rng.random(10), numba_func(nb_rng))


class TestRandomGenerators(MemoryLeakMixin, TestCase):
    def check_numpy_parity(self, distribution_func,
                           bitgen_type=None, seed=None,
                           test_size=None, test_dtype=None):

        distribution_func = numba.njit(distribution_func)
        if seed is None:
            seed = 1
        if bitgen_type is None:
            numba_rng_instance = np.random.default_rng(seed=seed)
            numpy_rng_instance = np.random.default_rng(seed=seed)
        else:
            numba_rng_instance = Generator(bitgen_type(seed))
            numpy_rng_instance = Generator(bitgen_type(seed))

        # Check parity for different size cases
        numba_res = distribution_func(numba_rng_instance,
                                      test_size, test_dtype)
        numpy_res = distribution_func.py_func(numpy_rng_instance,
                                              test_size, test_dtype)

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

    def test_npgen_boxing_refcount(self):
        rng_instance = np.random.default_rng()
        no_box = numba.njit(lambda x:x.random())
        do_box = numba.njit(lambda x:x)

        y = do_box(rng_instance)
        ref_1 = sys.getrefcount(rng_instance)
        del y
        no_box(rng_instance)
        ref_2 = sys.getrefcount(rng_instance)

        self.assertEqual(ref_1, ref_2 + 1)

    def test_bitgen_funcs(self):
        func_names = ["next_uint32", "next_uint64", "next_double"]
        funcs = [next_uint32, next_uint64, next_double]

        for _func, _func_name in zip(funcs, func_names):
            with self.subTest(_func=_func, _func_name=_func_name):
                self._test_bitgen_func_parity(_func_name, _func)

    def test_random(self):
        test_sizes = [None, (), (100,), (10, 20, 30)]
        test_dtypes = [np.float32, np.float64]
        bitgen_types = [None, MT19937]

        # Test with no arguments
        dist_func = lambda x, size, dtype:x.random()
        with self.subTest():
            # Provide single values so this test would run exactly once
            self.check_numpy_parity(dist_func, test_size=100,
                                    test_dtype=np.float64)

        dist_func = lambda x, size, dtype:x.random(size=size, dtype=dtype)

        for _size in test_sizes:
            for _dtype in test_dtypes:
                for _bitgen in bitgen_types:
                    with self.subTest(_size=_size, _dtype=_dtype,
                                      _bitgen=_bitgen):
                        self.check_numpy_parity(dist_func, _bitgen,
                                                None, _size, _dtype)


class TestGeneratorCaching(TestCase, SerialMixin):
    def test_randomgen_caching(self):
        nb_rng = np.random.default_rng(1)
        np_rng = np.random.default_rng(1)

        numba_func = numba.njit(lambda x: x.random(10), cache=True)
        self.assertPreciseEqual(np_rng.random(10), numba_func(nb_rng))
        # Run the function twice to make sure caching doesn't break anything.
        self.assertPreciseEqual(np_rng.random(10), numba_func(nb_rng))
        # Check that the function can be retrieved successfully from the cache.
        res = run_in_new_process_caching(test_generator_caching)
        self.assertEqual(res['exitcode'], 0)
