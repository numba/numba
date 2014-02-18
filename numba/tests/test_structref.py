from __future__ import print_function
import numba.unittest_support as unittest
from numba import types
from numba.compiler import compile_isolated
import numpy as np



def foo(a):
    b = a[0]
    a[0] = 123
    return b


class TestStructRef(unittest.TestCase):
    def test_complex(self):
        pyfunc = foo
        aryty = types.Array(types.complex128, 1, 'A')
        cres = compile_isolated(pyfunc, [aryty])
        a = np.array([321], dtype='complex128')
        a0 = a[0]
        cfunc = cres.entry_point
        self.assertEqual(cfunc(a), a0)
        self.assertEqual(a[0], 123)


if __name__ == '__main__':
    unittest.main()

