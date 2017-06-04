from __future__ import print_function, absolute_import
import numpy as np
from numba import ocl
from numba.ocl.testing import unittest


class TestOclComplex(unittest.TestCase):
    def test_ocl_complex_arg(self):
        @ocl.jit('void(complex128[:], complex128)')
        def foo(a, b):
            i = ocl.grid(1)
            a[i] += b


        a = np.arange(5, dtype=np.complex128)
        a0 = a.copy()
        foo[1, a.shape](a, 2j)
        self.assertTrue(np.allclose(a, a0 + 2j))


if __name__ == '__main__':
    unittest.main()


