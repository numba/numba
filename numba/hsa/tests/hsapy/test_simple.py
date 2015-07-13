from __future__ import print_function, absolute_import

import numpy as np
from numba import hsa
import numba.unittest_support as unittest


class TestSimple(unittest.TestCase):
    def test_array_access(self):
        @hsa.jit
        def udt(output):
            output[0] = 123

        out = np.zeros(1, dtype=np.intp)
        udt[1, 1](out)

        print(out)


if __name__ == '__main__':
    unittest.main()
