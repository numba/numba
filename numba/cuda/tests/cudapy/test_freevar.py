from __future__ import print_function, absolute_import

import numpy as np

from numba import cuda
from numba.cuda.testing import unittest, SerialMixin


class TestFreeVar(SerialMixin, unittest.TestCase):
    def test_freevar(self):
        """Make sure we can compile the following kernel with freevar reference
        in macros
        """
        from numba import float32

        size = 1024
        nbtype = float32
        @cuda.jit("(float32[::1], intp)")
        def foo(A, i):
            "Dummy function"
            sdata = cuda.shared.array(size,   # size is freevar
                                      dtype=nbtype)  # nbtype is freevar
            A[i] = sdata[i]

        A = np.arange(2, dtype="float32")
        foo(A, 0)


if __name__ == '__main__':
    unittest.main()
