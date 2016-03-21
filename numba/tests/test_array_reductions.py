from __future__ import division

from itertools import product

import numpy as np

from numba import unittest_support as unittest
from numba import jit, typeof
from numba.compiler import compile_isolated
from .support import TestCase, MemoryLeakMixin, tag


def array_all(arr):
    return arr.all()

def array_all_global(arr):
    return np.all(arr)

def array_any(arr):
    return arr.any()

def array_any_global(arr):
    return np.any(arr)

def array_cumprod(arr):
    return arr.cumprod()

def array_cumprod_global(arr):
    return np.cumprod(arr)

def array_cumsum(arr):
    return arr.cumsum()

def array_cumsum_global(arr):
    return np.cumsum(arr)

def array_sum(arr):
    return arr.sum()

def array_sum_global(arr):
    return np.sum(arr)

def array_prod(arr):
    return arr.prod()

def array_prod_global(arr):
    return np.prod(arr)

def array_mean(arr):
    return arr.mean()

def array_mean_global(arr):
    return np.mean(arr)

def array_var(arr):
    return arr.var()

def array_var_global(arr):
    return np.var(arr)

def array_std(arr):
    return arr.std()

def array_std_global(arr):
    return np.std(arr)

def array_min(arr):
    return arr.min()

def array_min_global(arr):
    return np.min(arr)

def array_max(arr):
    return arr.max()

def array_max_global(arr):
    return np.max(arr)

def array_argmin(arr):
    return arr.argmin()

def array_argmin_global(arr):
    return np.argmin(arr)

def array_argmax(arr):
    return arr.argmax()

def array_argmax_global(arr):
    return np.argmax(arr)

def array_median_global(arr):
    return np.median(arr)


def base_test_arrays(dtype):
    if dtype == np.bool_:
        def factory(n):
            assert n % 2 == 0
            return np.bool_([0, 1] * (n // 2))
    else:
        def factory(n):
            return np.arange(n, dtype=dtype) + 1

    a1 = factory(10)
    a2 = factory(10).reshape(2, 5)
    # The prod() of this array fits in a 32-bit int
    a3 = (factory(12))[::-1].reshape((2, 3, 2), order='A')
    assert not (a3.flags.c_contiguous or a3.flags.f_contiguous)

    return [a1, a2, a3]

def full_test_arrays(dtype):
    array_list = base_test_arrays(dtype)

    # Add floats with some mantissa
    if dtype == np.float32:
        array_list += [a / 10 for a in array_list]

    for a in array_list:
        assert a.dtype == np.dtype(dtype)
    return array_list

def run_comparative(compare_func, test_array):
    arrty = typeof(test_array)
    cres = compile_isolated(compare_func, [arrty])
    numpy_result = compare_func(test_array)
    numba_result = cres.entry_point(test_array)

    return numpy_result, numba_result


class TestArrayReductions(MemoryLeakMixin, TestCase):
    """
    Test array reduction methods and functions such as .sum(), .max(), etc.
    """

    @tag('important')
    def test_all_basic(self, pyfunc=array_all):
        cfunc = jit(nopython=True)(pyfunc)
        def check(arr):
            self.assertPreciseEqual(pyfunc(arr), cfunc(arr))

        arr = np.float64([1.0, 0.0, float('inf'), float('nan')])
        check(arr)
        arr[1] = -0.0
        check(arr)
        arr[1] = 1.5
        check(arr)
        arr = arr.reshape((2, 2))
        check(arr)
        check(arr[::-1])

    @tag('important')
    def test_any_basic(self, pyfunc=array_any):
        cfunc = jit(nopython=True)(pyfunc)
        def check(arr):
            self.assertPreciseEqual(pyfunc(arr), cfunc(arr))

        arr = np.float64([0.0, -0.0, 0.0, 0.0])
        check(arr)
        arr[2] = float('nan')
        check(arr)
        arr[2] = float('inf')
        check(arr)
        arr[2] = 1.5
        check(arr)
        arr = arr.reshape((2, 2))
        check(arr)
        check(arr[::-1])

    @tag('important')
    def test_sum_basic(self):
        arr = np.arange(100)
        npr, nbr = run_comparative(array_sum, arr)
        self.assertPreciseEqual(npr, nbr)

    @tag('important')
    def test_mean_basic(self):
        arr = np.arange(100)
        npr, nbr = run_comparative(array_mean, arr)
        self.assertPreciseEqual(npr, nbr, prec="double")

    @tag('important')
    def test_var_basic(self):
        arr = np.arange(100)
        npr, nbr = run_comparative(array_var, arr)
        self.assertPreciseEqual(npr, nbr, prec="double")

    @tag('important')
    def test_std_basic(self):
        arr = np.arange(100)
        npr, nbr = run_comparative(array_std, arr)
        self.assertPreciseEqual(npr, nbr, prec="double")

    @tag('important')
    def test_min_basic(self):
        arr = np.arange(100)
        npr, nbr = run_comparative(array_min, arr)
        self.assertPreciseEqual(npr, nbr)

    @tag('important')
    def test_max_basic(self):
        arr = np.arange(100)
        npr, nbr = run_comparative(array_max, arr)
        self.assertPreciseEqual(npr, nbr)

    @tag('important')
    def test_argmin_basic(self):
        arr = np.arange(100)
        npr, nbr = run_comparative(array_argmin, arr)
        self.assertPreciseEqual(npr, nbr)

    @tag('important')
    def test_argmax_basic(self):
        arr = np.arange(100)
        npr, nbr = run_comparative(array_argmax, arr)
        self.assertPreciseEqual(npr, nbr)

    def test_median_odd(self):
        arr = np.arange(101)
        np.random.shuffle(arr)
        npr, nbr = run_comparative(array_median_global, arr)
        self.assertPreciseEqual(npr, nbr)

    def test_median_even(self):
        arr = np.arange(100)
        np.random.shuffle(arr)
        npr, nbr = run_comparative(array_median_global, arr)
        self.assertPreciseEqual(npr, nbr)

    def test_median_sorted(self):
        arr = np.arange(100)
        npr, nbr = run_comparative(array_median_global, arr)
        self.assertPreciseEqual(npr, nbr)

    def test_median_revsorted(self):
        arr = np.arange(100, 0, -1)
        npr, nbr = run_comparative(array_median_global, arr)
        self.assertPreciseEqual(npr, nbr)

    def test_median_duplicate(self):
        arr = np.ones(100)
        npr, nbr = run_comparative(array_median_global, arr)
        self.assertPreciseEqual(npr, nbr)

    def test_array_sum_global(self):
        arr = np.arange(10, dtype=np.int32)
        arrty = typeof(arr)
        self.assertEqual(arrty.ndim, 1)
        self.assertEqual(arrty.layout, 'C')

        cres = compile_isolated(array_sum_global, [arrty])
        cfunc = cres.entry_point

        self.assertEqual(np.sum(arr), cfunc(arr))

    def test_array_prod_int_1d(self):
        arr = np.arange(10, dtype=np.int32) + 1
        arrty = typeof(arr)
        self.assertEqual(arrty.ndim, 1)
        self.assertEqual(arrty.layout, 'C')

        cres = compile_isolated(array_prod, [arrty])
        cfunc = cres.entry_point

        self.assertEqual(arr.prod(), cfunc(arr))

    def test_array_prod_float_1d(self):
        arr = np.arange(10, dtype=np.float32) + 1 / 10
        arrty = typeof(arr)
        self.assertEqual(arrty.ndim, 1)
        self.assertEqual(arrty.layout, 'C')

        cres = compile_isolated(array_prod, [arrty])
        cfunc = cres.entry_point

        np.testing.assert_allclose(arr.prod(), cfunc(arr))

    def test_array_prod_global(self):
        arr = np.arange(10, dtype=np.int32)
        arrty = typeof(arr)
        self.assertEqual(arrty.ndim, 1)
        self.assertEqual(arrty.layout, 'C')

        cres = compile_isolated(array_prod_global, [arrty])
        cfunc = cres.entry_point

        np.testing.assert_allclose(np.prod(arr), cfunc(arr))

    def check_cumulative(self, pyfunc):
        arr = np.arange(2, 10, dtype=np.int16)
        expected, got = run_comparative(pyfunc, arr)
        self.assertPreciseEqual(got, expected)
        arr = np.linspace(2, 8, 6)
        expected, got = run_comparative(pyfunc, arr)
        self.assertPreciseEqual(got, expected)
        arr = arr.reshape((3, 2))
        expected, got = run_comparative(pyfunc, arr)
        self.assertPreciseEqual(got, expected)

    @tag('important')
    def test_array_cumsum(self):
        self.check_cumulative(array_cumsum)

    def test_array_cumsum_global(self):
        self.check_cumulative(array_cumsum_global)

    @tag('important')
    def test_array_cumprod(self):
        self.check_cumulative(array_cumprod)

    def test_array_cumprod_global(self):
        self.check_cumulative(array_cumprod_global)

    def check_aggregation_magnitude(self, pyfunc, is_prod=False):
        """
        Check that integer overflows are avoided (issue #931).
        """
        # Overflows are avoided here (ints are cast either to intp
        # or float64).
        n_items = 2 if is_prod else 10  # avoid overflow on prod()
        arr = (np.arange(n_items) + 40000).astype('int16')
        npr, nbr = run_comparative(pyfunc, arr)
        self.assertPreciseEqual(npr, nbr)
        # Overflows are avoided for functions returning floats here.
        # Other functions may wrap around.
        arr = (np.arange(10) + 2**60).astype('int64')
        npr, nbr = run_comparative(pyfunc, arr)
        self.assertPreciseEqual(npr, nbr)
        arr = arr.astype('uint64')
        npr, nbr = run_comparative(pyfunc, arr)
        self.assertPreciseEqual(npr, nbr)

    def test_sum_magnitude(self):
        self.check_aggregation_magnitude(array_sum)
        self.check_aggregation_magnitude(array_sum_global)

    def test_cumsum_magnitude(self):
        self.check_aggregation_magnitude(array_cumsum)
        self.check_aggregation_magnitude(array_cumsum_global)

    def test_prod_magnitude(self):
        self.check_aggregation_magnitude(array_prod, is_prod=True)
        self.check_aggregation_magnitude(array_prod_global, is_prod=True)

    def test_cumprod_magnitude(self):
        self.check_aggregation_magnitude(array_cumprod, is_prod=True)
        self.check_aggregation_magnitude(array_cumprod_global, is_prod=True)

    def test_mean_magnitude(self):
        self.check_aggregation_magnitude(array_mean)
        self.check_aggregation_magnitude(array_mean_global)

    def test_var_magnitude(self):
        self.check_aggregation_magnitude(array_var)
        self.check_aggregation_magnitude(array_var_global)

    def test_std_magnitude(self):
        self.check_aggregation_magnitude(array_std)
        self.check_aggregation_magnitude(array_std_global)

    def _do_check_nptimedelta(self, pyfunc, arr):
        arrty = typeof(arr)

        cres = compile_isolated(pyfunc, [arrty])
        cfunc = cres.entry_point

        self.assertPreciseEqual(cfunc(arr), pyfunc(arr))
        # Even vs. odd size, for np.median
        self.assertPreciseEqual(cfunc(arr[:-1]), pyfunc(arr[:-1]))
        # Test with different orders, for np.median
        arr = arr[::-1].copy()  # Keep 'C' layout
        self.assertPreciseEqual(cfunc(arr), pyfunc(arr))
        np.random.shuffle(arr)
        self.assertPreciseEqual(cfunc(arr), pyfunc(arr))
        # Test with a NaT
        arr[arr.size // 2] = 'NaT'
        self.assertPreciseEqual(cfunc(arr), pyfunc(arr))
        # Test with all NaTs
        arr.fill(arrty.dtype('NaT'))
        self.assertPreciseEqual(cfunc(arr), pyfunc(arr))

    def check_npdatetime(self, pyfunc):
        arr = np.arange(10).astype(dtype='M8[Y]')
        self._do_check_nptimedelta(pyfunc, arr)

    def check_nptimedelta(self, pyfunc):
        arr = np.arange(10).astype(dtype='m8[s]')
        self._do_check_nptimedelta(pyfunc, arr)

    def test_min_npdatetime(self):
        self.check_npdatetime(array_min)
        self.check_nptimedelta(array_min)

    def test_max_npdatetime(self):
        self.check_npdatetime(array_max)
        self.check_nptimedelta(array_max)

    def test_argmin_npdatetime(self):
        self.check_npdatetime(array_argmin)
        self.check_nptimedelta(array_argmin)

    def test_argmax_npdatetime(self):
        self.check_npdatetime(array_argmax)
        self.check_nptimedelta(array_argmax)

    def test_median_npdatetime(self):
        self.check_nptimedelta(array_median_global)

    def test_sum_npdatetime(self):
        self.check_nptimedelta(array_sum)

    def test_cumsum_npdatetime(self):
        self.check_nptimedelta(array_cumsum)

    def test_mean_npdatetime(self):
        self.check_nptimedelta(array_mean)

    @classmethod
    def install_generated_tests(cls):
        # These form a testing product where each of the combinations are tested
        reduction_funcs = [array_sum, array_sum_global,
                           array_prod, array_prod_global,
                           array_mean, array_mean_global,
                           array_var, array_var_global,
                           array_std, array_std_global,
                           array_min, array_min_global,
                           array_max, array_max_global,
                           array_argmin, array_argmin_global,
                           array_argmax, array_argmax_global,
                           array_all, array_all_global,
                           array_any, array_any_global,
                           ]
        dtypes_to_test = [np.int32, np.float32, np.bool_]

        # Install tests on class
        for dt in dtypes_to_test:
            test_arrays = full_test_arrays(dt)
            for red_func, test_array in product(reduction_funcs, test_arrays):
                # Create the name for the test function
                test_name = "test_{0}_{1}_{2}d".format(red_func.__name__, test_array.dtype.name, test_array.ndim)

                def new_test_function(self, redFunc=red_func, testArray=test_array, testName=test_name):
                    npr, nbr = run_comparative(redFunc, testArray)
                    self.assertPreciseEqual(npr, nbr, msg=test_name, prec="single")

                # Install it into the class
                setattr(cls, test_name, new_test_function)


TestArrayReductions.install_generated_tests()


if __name__ == '__main__':
    unittest.main()
