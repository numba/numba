from __future__ import print_function, absolute_import, division
import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, SerialMixin
from numba.cuda.testing import skip_on_cudasim


@skip_on_cudasim('Simulator does not have definitions attribute')
class TestCudaAutoJit(SerialMixin, unittest.TestCase):
    def test_autojit(self):
        @cuda.autojit
        def what(a, b, c):
            pass

        what(np.empty(1), 1.0, 21)
        what(np.empty(1), 1.0, 21)
        what(np.empty(1), np.empty(1, dtype=np.int32), 21)
        what(np.empty(1), np.empty(1, dtype=np.int32), 21)
        what(np.empty(1), 1.0, 21)

        self.assertTrue(len(what.definitions) == 2)


if __name__ == '__main__':
    unittest.main()

