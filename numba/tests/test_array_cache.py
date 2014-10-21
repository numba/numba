import numba.unittest_support as unittest
import numpy as np
from numba import types, typing, typeof
from numba.compiler import compile_isolated
from numba.targets.arrayobj import array_sum, array_prod
from numpy.lib.stride_tricks import as_strided

def array_sum_fn(arr):
    return arr.sum()

def array_prod_fn(arr):
    return arr.prod()

class TestArrayCaching(unittest.TestCase):
    
    def generate_cache_tests(ns, cases):
        for test_name, arr, layout, red_fn, ref_fn in cases:
            def test_method(self, arr=arr, layout=layout, red_fn=red_fn,
                            ref_fn=ref_fn):
           
                arrty = typeof(arr)
                self.assertEqual(arrty.ndim, 1)
                self.assertEqual(arrty.layout, layout)

                cres = compile_isolated(red_fn, [arrty])

                expectedArg = types.Array(types.int32, 1, layout, const=False)
                expectedSig = typing.signature(types.int32, expectedArg)
                expectedKey = (ref_fn, expectedSig, types.int32)
                keys = list(cres.target_context.cached_internal_func)
                self.assertEqual(1, len(keys))
                actualKey = keys[0]
                self.assertEqual(expectedKey, actualKey)

            test_method.__name__ = test_name
            ns[test_name] = test_method

    flatarr = np.arange(10, dtype=np.int32)
    stridedarr = as_strided(flatarr, strides=(8,), shape=(5,))
    
    generate_cache_tests(locals(),
        (('test_flat_sum_key', flatarr, 'C', array_sum_fn, array_sum),
         ('test_flat_prod_key', flatarr, 'C', array_prod_fn, array_prod),
         ('test_strided_sum_key', stridedarr, 'A', array_sum_fn, array_sum),
         ('test_strided_prod_key', stridedarr, 'A', array_prod_fn, array_prod)))

if __name__ == '__main__':
    unittest.main()

