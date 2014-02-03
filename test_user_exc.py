from numba.compiler import compile_isolated
from numba import types
from numba import unittest_support as unittest
from numba.targets.cpu import NativeError
import numpy as np


class TestUserExc(unittest.TestCase):
    def test_unituple_index_error(self):
        def pyfunc(a, i):
            return a.shape[i]

        cres = compile_isolated(pyfunc, (types.Array(types.int32, 1, 'A'),
                                         types.int32))

        cfunc = cres.entry_point
        a = np.empty(2)

        self.assertEqual(cfunc(a, 0), pyfunc(a, 0))

        with self.assertRaises(NativeError):
            cfunc(a, 2)


if __name__ == '__main__':
    unittest.main()
