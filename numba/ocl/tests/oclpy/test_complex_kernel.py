from __future__ import print_function, absolute_import
import numpy as np
from numba import ocl
from numba.ocl.testing import unittest
from numba.ocl.testing import OCLTestCase


class TestOclComplex(OCLTestCase):
    def test_ocl_complex_arg(self):
        @ocl.jit
        def foo(a, b):
            i = ocl.get_global_id(0)
            a[i] += b


        a = np.arange(5, dtype=np.complex128)
        a0 = a.copy()
        foo[1, a.shape](a, 2j)
        self.assertTrue(np.allclose(a, a0 + 2j))


if __name__ == '__main__':
    unittest.main()


