from __future__ import division

from itertools import product

import numpy as np

from numba import unittest_support as unittest
from numba import typeof, types
from numba.compiler import compile_isolated
from .support import TestCase


def array_flat(arr, out):
    for i, v in enumerate(arr.flat):
        out[i] = v

def array_flat_sum(arr):
    s = 0
    for i, v in enumerate(arr.flat):
        s = s + (i + 1) * v
    return s

def array_ndenumerate_sum(arr):
    s = 0
    for (i, j), v in np.ndenumerate(arr):
        s = s + (i + 1) * (j + 1) * v
    return s

def np_ndindex(x, y):
    s = 0
    n = 0
    for i, j in np.ndindex(x, y):
        s = s + (i + 1) * (j + 1)
    return s

def np_ndindex_array(arr):
    s = 0
    n = 0
    for indices in np.ndindex(arr.shape):
        for i, j in enumerate(indices):
            s = s + (i + 1) * (j + 1)
    return s


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


def base_test_arrays(dtype):
    a1 = np.arange(10, dtype=dtype) + 1
    a2 = np.arange(10, dtype=dtype).reshape(2, 5) + 1
    a3 = (np.arange(60, dtype=dtype))[::2].reshape((2, 5, 3), order='A')

    return [a1, a2, a3]


def yield_test_props():
    return [(1, 'C'), (2, 'C'), (3, 'A')]


def full_test_arrays(dtype):
    array_list = base_test_arrays(dtype)

    #Add floats with some mantissa
    if dtype == np.float32:
        array_list += [a / 10 for a in array_list]

    return array_list


def run_comparative(compare_func, test_array):
    arrty = typeof(test_array)
    cres = compile_isolated(compare_func, [arrty])
    numpy_result = compare_func(test_array)
    numba_result = cres.entry_point(test_array)

    return numpy_result, numba_result


def array_prop(aray):
    arrty = typeof(aray)
    return (arrty.ndim, arrty.layout)


class TestArrayMethods(TestCase):
    def test_array_ndim_and_layout(self):
        for testArray, testArrayProps in zip(base_test_arrays(np.int32), yield_test_props()):
            self.assertEqual(array_prop(testArray), testArrayProps)

    def test_sum_basic(self):
        arr = np.arange(100)
        npr, nbr = run_comparative(array_sum, arr)
        self.assertPreciseEqual(npr, nbr)

    def test_mean_basic(self):
        arr = np.arange(100)
        npr, nbr = run_comparative(array_mean, arr)
        self.assertPreciseEqual(npr, nbr, prec="double")

    def test_var_basic(self):
        arr = np.arange(100)
        npr, nbr = run_comparative(array_var, arr)
        self.assertPreciseEqual(npr, nbr, prec="double")

    def test_std_basic(self):
        arr = np.arange(100)
        npr, nbr = run_comparative(array_std, arr)
        self.assertPreciseEqual(npr, nbr, prec="double")

    def test_min_basic(self):
        arr = np.arange(100)
        npr, nbr = run_comparative(array_min, arr)
        self.assertPreciseEqual(npr, nbr)

    def test_max_basic(self):
        arr = np.arange(100)
        npr, nbr = run_comparative(array_max, arr)
        self.assertPreciseEqual(npr, nbr)

    def test_argmin_basic(self):
        arr = np.arange(100)
        npr, nbr = run_comparative(array_argmin, arr)
        self.assertPreciseEqual(npr, nbr)

    def test_argmax_basic(self):
        arr = np.arange(100)
        npr, nbr = run_comparative(array_argmax, arr)
        self.assertPreciseEqual(npr, nbr)

    def check_array_flat(self, arr, arrty=None):
        out = np.zeros(arr.size, dtype=arr.dtype)
        nb_out = out.copy()
        if arrty is None:
            arrty = typeof(arr)

        cres = compile_isolated(array_flat, [arrty, typeof(out)])
        cfunc = cres.entry_point

        array_flat(arr, out)
        cfunc(arr, nb_out)

        self.assertTrue(np.all(out == nb_out), (out, nb_out))

    def check_array_unary(self, arr, arrty, func):
        cres = compile_isolated(func, [arrty])
        cfunc = cres.entry_point
        self.assertPreciseEqual(cfunc(arr), func(arr))

    def check_array_flat_sum(self, arr, arrty):
        self.check_array_unary(arr, arrty, array_flat_sum)

    def check_array_ndenumerate_sum(self, arr, arrty):
        self.check_array_unary(arr, arrty, array_ndenumerate_sum)

    def test_array_sum_int_1d(self):
        arr = np.arange(10, dtype=np.int32)
        arrty = typeof(arr)
        self.assertEqual(arrty.ndim, 1)
        self.assertEqual(arrty.layout, 'C')

    def test_array_flat_3d(self):
        arr = np.arange(24).reshape(4, 2, 3)

        arrty = typeof(arr)
        self.assertEqual(arrty.ndim, 3)
        self.assertEqual(arrty.layout, 'C')
        self.assertTrue(arr.flags.c_contiguous)
        # Test with C-contiguous array
        self.check_array_flat(arr)
        # Test with Fortran-contiguous array
        arr = arr.transpose()
        self.assertFalse(arr.flags.c_contiguous)
        self.assertTrue(arr.flags.f_contiguous)
        self.assertEqual(typeof(arr).layout, 'F')
        self.check_array_flat(arr)
        # Test with non-contiguous array
        arr = arr[::2]
        self.assertFalse(arr.flags.c_contiguous)
        self.assertFalse(arr.flags.f_contiguous)
        self.assertEqual(typeof(arr).layout, 'A')
        self.check_array_flat(arr)

    def test_array_flat_empty(self):
        # Test .flat() with various shapes of empty arrays, contiguous
        # and non-contiguous (see issue #846).
        arr = np.zeros(0, dtype=np.int32)
        arr = arr.reshape(0, 2)
        arrty = types.Array(types.int32, 2, layout='C')
        self.check_array_flat_sum(arr, arrty)
        arrty = types.Array(types.int32, 2, layout='F')
        self.check_array_flat_sum(arr, arrty)
        arrty = types.Array(types.int32, 2, layout='A')
        self.check_array_flat_sum(arr, arrty)
        arr = arr.reshape(2, 0)
        arrty = types.Array(types.int32, 2, layout='C')
        self.check_array_flat_sum(arr, arrty)
        arrty = types.Array(types.int32, 2, layout='F')
        self.check_array_flat_sum(arr, arrty)
        arrty = types.Array(types.int32, 2, layout='A')
        self.check_array_flat_sum(arr, arrty)

    def test_array_ndenumerate_2d(self):
        arr = np.arange(12).reshape(4, 3)
        arrty = typeof(arr)
        self.assertEqual(arrty.ndim, 2)
        self.assertEqual(arrty.layout, 'C')
        self.assertTrue(arr.flags.c_contiguous)
        # Test with C-contiguous array
        self.check_array_ndenumerate_sum(arr, arrty)
        # Test with Fortran-contiguous array
        arr = arr.transpose()
        self.assertFalse(arr.flags.c_contiguous)
        self.assertTrue(arr.flags.f_contiguous)
        arrty = typeof(arr)
        self.assertEqual(arrty.layout, 'F')
        self.check_array_ndenumerate_sum(arr, arrty)
        # Test with non-contiguous array
        arr = arr[::2]
        self.assertFalse(arr.flags.c_contiguous)
        self.assertFalse(arr.flags.f_contiguous)
        arrty = typeof(arr)
        self.assertEqual(arrty.layout, 'A')
        self.check_array_ndenumerate_sum(arr, arrty)

    def test_array_ndenumerate_empty(self):
        arr = np.zeros(0, dtype=np.int32)
        arr = arr.reshape(0, 2)
        arrty = types.Array(types.int32, 2, layout='C')
        self.check_array_ndenumerate_sum(arr, arrty)
        arrty = types.Array(types.int32, 2, layout='F')
        self.check_array_ndenumerate_sum(arr, arrty)
        arrty = types.Array(types.int32, 2, layout='A')
        self.check_array_ndenumerate_sum(arr, arrty)
        arr = arr.reshape(2, 0)
        arrty = types.Array(types.int32, 2, layout='C')
        self.check_array_flat_sum(arr, arrty)
        arrty = types.Array(types.int32, 2, layout='F')
        self.check_array_flat_sum(arr, arrty)
        arrty = types.Array(types.int32, 2, layout='A')
        self.check_array_flat_sum(arr, arrty)

    def test_np_ndindex(self):
        func = np_ndindex
        cres = compile_isolated(func, [types.int32, types.int32])
        cfunc = cres.entry_point
        self.assertPreciseEqual(cfunc(3, 4), func(3, 4))
        self.assertPreciseEqual(cfunc(3, 0), func(3, 0))
        self.assertPreciseEqual(cfunc(0, 3), func(0, 3))
        self.assertPreciseEqual(cfunc(0, 0), func(0, 0))

    def test_np_ndindex_array(self):
        func = np_ndindex_array
        arr = np.arange(12, dtype=np.int32)
        self.check_array_unary(arr, typeof(arr), func)
        arr = arr.reshape((4, 3))
        self.check_array_unary(arr, typeof(arr), func)
        arr = arr.reshape((2, 2, 3))
        self.check_array_unary(arr, typeof(arr), func)

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

    def test_prod_magnitude(self):
        self.check_aggregation_magnitude(array_prod, is_prod=True)
        self.check_aggregation_magnitude(array_prod_global, is_prod=True)

    def test_mean_magnitude(self):
        self.check_aggregation_magnitude(array_mean)
        self.check_aggregation_magnitude(array_mean_global)

    def test_var_magnitude(self):
        self.check_aggregation_magnitude(array_var)
        self.check_aggregation_magnitude(array_var_global)

    def test_std_magnitude(self):
        self.check_aggregation_magnitude(array_std)
        self.check_aggregation_magnitude(array_std_global)


# These form a testing product where each of the combinations are tested
reduction_funcs = [array_sum, array_sum_global,
                   array_prod, array_prod_global,
                   array_mean, array_mean_global,
                   array_var, array_var_global,
                   array_std, array_std_global,
                   array_min, array_min_global,
                   array_max, array_max_global,
                   array_argmin, array_argmin_global,
                   array_argmax, array_argmax_global]
dtypes_to_test = [np.int32, np.float32]

# Install tests on class above
for dt in dtypes_to_test:
    for red_func, test_array in product(reduction_funcs, full_test_arrays(dt)):
        # Create the name for the test function
        test_name = "test_{0}_{1}_{2}d".format(red_func.__name__, test_array.dtype.name, test_array.ndim)

        def new_test_function(self, redFunc=red_func, testArray=test_array, testName=test_name):
            npr, nbr = run_comparative(redFunc, testArray)
            self.assertPreciseEqual(npr, nbr, msg=test_name, prec="single")

        # Install it into the class
        setattr(TestArrayMethods, test_name, new_test_function)


if __name__ == '__main__':
    unittest.main()
