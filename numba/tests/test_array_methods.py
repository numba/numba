from __future__ import division

from itertools import product

import numpy as np

from numba import unittest_support as unittest
from numba import typeof, types
from numba.compiler import compile_isolated
from numba.numpy_support import as_dtype
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


def np_around_array(arr, decimals, out):
    np.around(arr, decimals, out)

def np_around_binary(val, decimals):
    return np.around(val, decimals)

def np_around_unary(val):
    return np.around(val)

def np_round_array(arr, decimals, out):
    np.round(arr, decimals, out)

def np_round_binary(val, decimals):
    return np.round(val, decimals)

def np_round_unary(val):
    return np.round(val)

def _fixed_np_round(arr, decimals=0, out=None):
    """
    A slightly bugfixed version of np.round().
    """
    if out is not None and arr.dtype.kind == 'c':
        # workaround for https://github.com/numpy/numpy/issues/5779
        _fixed_np_round(arr.real, decimals, out.real)
        _fixed_np_round(arr.imag, decimals, out.imag)
        return out
    else:
        res = np.round(arr, decimals, out)
        if out is None:
            # workaround for https://github.com/numpy/numpy/issues/5780
            def fixup_signed_zero(arg, res):
                if res == 0.0 and arg < 0:
                    return -np.abs(res)
                else:
                    return res
            if isinstance(arr, (complex, np.complexfloating)):
                res = complex(fixup_signed_zero(arr.real, res.real),
                              fixup_signed_zero(arr.imag, res.imag))
            else:
                res = fixup_signed_zero(arr, res)
        return res


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

    def test_array_cumsum(self):
        self.check_cumulative(array_cumsum)

    def test_array_cumsum_global(self):
        self.check_cumulative(array_cumsum_global)

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

    def check_round_array(self, pyfunc):
        def check_round(cfunc, values, inty, outty, decimals):
            # Create input and output arrays of the right type
            arr = values.astype(as_dtype(inty))
            out = np.zeros_like(arr).astype(as_dtype(outty))
            pyout = out.copy()
            _fixed_np_round(arr, decimals, pyout)
            cfunc(arr, decimals, out)
            np.testing.assert_allclose(out, pyout)
            # Output shape mismatch
            with self.assertRaises(ValueError) as raises:
                cfunc(arr, decimals, out[1:])
            self.assertEqual(str(raises.exception),
                             "invalid output shape")

        def check_types(argtypes, outtypes, values):
            for inty, outty in product(argtypes, outtypes):
                cres = compile_isolated(pyfunc,
                                        (types.Array(inty, 1, 'A'),
                                         types.int32,
                                         types.Array(outty, 1, 'A')))
                cfunc = cres.entry_point
                check_round(cres.entry_point, values, inty, outty, 0)
                check_round(cres.entry_point, values, inty, outty, 1)
                if not isinstance(outty, types.Integer):
                    check_round(cres.entry_point, values * 10, inty, outty, -1)
                else:
                    # Avoid Numpy bug when output is an int:
                    # https://github.com/numpy/numpy/issues/5777
                    pass

        values = np.array([-3.0, -2.5, -2.25, -1.5, 1.5, 2.25, 2.5, 2.75])

        argtypes = (types.float64, types.float32, types.int32)
        check_types(argtypes, argtypes, values)

        argtypes = (types.complex64, types.complex128)
        check_types(argtypes, argtypes, values * (1 - 1j))

    def test_round_array(self):
        self.check_round_array(np_round_array)

    def test_around_array(self):
        self.check_round_array(np_around_array)

    def check_round_scalar(self, unary_pyfunc, binary_pyfunc):
        base_values = [-3.0, -2.5, -2.25, -1.5, 1.5, 2.25, 2.5, 2.75]
        complex_values = [x * (1 - 1j) for x in base_values]
        int_values = [int(x) for x in base_values]
        argtypes = (types.float64, types.float32, types.int32,
                    types.complex64, types.complex128)
        argvalues = [base_values, base_values, int_values,
                     complex_values, complex_values]

        pyfunc = binary_pyfunc
        for ty, values in zip(argtypes, argvalues):
            cres = compile_isolated(pyfunc, (ty, types.int32))
            cfunc = cres.entry_point
            for decimals in (1, 0, -1):
                for v in values:
                    if decimals > 0:
                        v *= 10
                    expected = _fixed_np_round(v, decimals)
                    got = cfunc(v, decimals)
                    self.assertPreciseEqual(got, expected)

        pyfunc = unary_pyfunc
        for ty, values in zip(argtypes, argvalues):
            cres = compile_isolated(pyfunc, (ty,))
            cfunc = cres.entry_point
            for v in values:
                expected = _fixed_np_round(v)
                got = cfunc(v)
                self.assertPreciseEqual(got, expected)

    def test_round_scalar(self):
        self.check_round_scalar(np_round_unary, np_round_binary)

    def test_around_scalar(self):
        self.check_round_scalar(np_around_unary, np_around_binary)


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
