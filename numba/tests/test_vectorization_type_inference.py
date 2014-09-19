from __future__ import print_function
from numba import vectorize, jit, bool_, double, int_, float_, typeof, int8
import numba.unittest_support as unittest
import numpy as np


def add(a, b):
    return a + b


def func(dtypeA, dtypeB):
    A = np.arange(10, dtype=dtypeA)
    B = np.arange(10, dtype=dtypeB)
    return typeof(vector_add(A, B))


class TestVectTypeInfer(unittest.TestCase):
    
    @unittest.expectedFailure
    def test_type_inference(self):
        global vector_add
        vector_add = vectorize([
            bool_(double, int_),
            double(double, double),
            float_(double, float_),
        ])(add)

        cfunc = jit(func)

        self.assertEqual(cfunc(np.dtype(np.float64), np.dtype('i')), int8[:])
        self.assertEqual(cfunc(np.dtype(np.float64), np.dtype(np.float64)),
                         double[:])
        self.assertEqual(cfunc(np.dtype(np.float64), np.dtype(np.float32)),
                         float_[:])


if __name__ == '__main__':
    unittest.main()
