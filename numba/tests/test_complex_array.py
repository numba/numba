from __future__ import print_function, absolute_import
import numba.unittest_support as unittest
import numpy as np
from numba.compiler import compile_isolated
from numba import types
from .support import MemoryLeakMixin


def copy(a, b):
    for i in range(a.shape[0]):
        b[i] = a[i]


class TestArray(MemoryLeakMixin, unittest.TestCase):
    def test_copy_complex64(self):
        pyfunc = copy
        carray = types.Array(types.complex64, 1, "C")

        cres = compile_isolated(pyfunc, (carray, carray))
        cfunc = cres.entry_point

        a = np.arange(10, dtype="complex64") + 1j
        control = np.zeros_like(a)
        result = np.zeros_like(a)

        pyfunc(a, control)
        cfunc(a, result)

        self.assertTrue(np.all(control == result))





if __name__ == '__main__':
    unittest.main()
