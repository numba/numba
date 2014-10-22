from __future__ import division
from itertools import product
from numba import unittest_support as unittest
from numba import typeof
from numba.compiler import compile_isolated
import numpy as np



def array_sum(arr):
    return arr.sum()


def array_sum_global(arr):
    return np.sum(arr)


def array_prod(arr):
    return arr.prod()


def array_prod_global(arr):
    return np.prod(arr)


def array_flat(arr, out):
    for i, v in enumerate(arr.flat):
        out[i] = v


def array_mean(arr):
    return arr.mean()


def array_mean_global(arr):
    return np.mean(arr)


def base_test_arrays(dtype):
    a1 = np.arange(10, dtype=dtype) + 1
    a2 = np.arange(10, dtype=dtype).reshape(2, 5) + 1
    a3 = (np.arange(60, dtype=dtype))[::2].reshape((2, 5, 3), order='A')

    return [a1, a2, a3]


def full_test_arrays(dtype):
    array_list = base_test_arrays(dtype)

    #Add floats with some mantissa
    if dtype == np.float32:
        array_list += [a / 10 for a in array_list]

    return array_list


def yield_test_props():
    yield (1, 'C')
    yield (2, 'C')
    yield (3, 'A')

def run_comparative(funcToCompare, testArray):
    arrty = typeof(testArray)
    cres = compile_isolated(funcToCompare, [arrty])
    numpyResult = funcToCompare(testArray)
    numbaResult = cres.entry_point(testArray)

    return numpyResult, numbaResult


def array_prop(aray):
    arrty = typeof(aray)
    return (arrty.ndim, arrty.layout)
    

class TestArrayMethods(unittest.TestCase):
    def test_array_ndim_and_layout(self):
        for testArray, testArrayProps in zip(base_test_arrays(np.int32), yield_test_props()):
            self.assertEqual(array_prop(testArray), testArrayProps)


    def test_array_flat_3d(self):
        arr = np.arange(50).reshape(5, 2, 5)

        arrty = typeof(arr)
        self.assertEqual(arrty.ndim, 3)

        out = np.zeros(arr.size, dtype=arr.dtype)
        nb_out = out.copy()

        cres = compile_isolated(array_flat, [arrty, typeof(out)])
        cfunc = cres.entry_point

        array_flat(arr, out)
        cfunc(arr, nb_out)

        self.assertTrue(np.all(out == nb_out))

# These form a testing product where each of the combinations are tested
reduction_funcs = [array_sum, array_sum_global, 
                   array_prod, array_prod_global, 
                   array_mean, array_mean_global]
dtypes_to_test = [np.int32, np.float32]

# Install tests on class above
for dt in dtypes_to_test:
    for redFunc, testArray in product(reduction_funcs, full_test_arrays(dt)):
        # Create the name for the test function 
        testName = "test_{0}_{1}_{2}d".format(redFunc.__name__, testArray.dtype.name, testArray.ndim)

        arr = testArray.copy()
        def installedFunction(selfish):
            numpyResult, numbaResult = run_comparative(redFunc, arr)
            if numpyResult.dtype is np.int32:
                allEqual = np.all(numpyResult == numbaResult)
                self.assertTrue(allEqual)
            elif numpyResult.dtype is np.float32:
                allClose = np.allclose(numpyResult, numbaResult, rtol=1e-6)
                self.assertTrue(allClose)
        setattr(TestArrayMethods, testName, installedFunction)

if __name__ == '__main__':
    unittest.main()
