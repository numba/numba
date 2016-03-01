from __future__ import print_function
import pickle
import numpy as np
from numba import cuda
from numba import unittest_support as unittest
from numba.config import ENABLE_CUDASIM


@unittest.skipIf(ENABLE_CUDASIM, 'pickling not supported in CUDASIM')
class TestPickle(unittest.TestCase):
    def check_call(self, callee):
        # serialize and rebuild
        foo1 = pickle.loads(pickle.dumps(callee))

        # call rebuild function
        arr = np.array([100])
        foo1(arr)
        self.assertEqual(arr[0], 101)

        # test serialization of previously serialized object
        foo2 = pickle.loads(pickle.dumps(foo1))
        # call rebuild function
        foo2(arr)
        self.assertEqual(arr[0], 102)

        # test propagation of thread, block config
        foo3 = pickle.loads(pickle.dumps(foo2[5, 8]))
        self.assertEqual(foo3.griddim, (5, 1, 1))
        self.assertEqual(foo3.blockdim, (8, 1, 1))

    def test_pickling_jit(self):
        @cuda.jit(device=True)
        def inner(a):
            return a + 1

        @cuda.jit('void(intp[:])')
        def foo(arr):
            arr[0] = inner(arr[0])

        self.check_call(foo)

    def test_pickling_autojit(self):

        @cuda.jit(device=True)
        def inner(a):
            return a + 1

        @cuda.jit
        def foo(arr):
            arr[0] = inner(arr[0])


if __name__ == '__main__':
    unittest.main()
