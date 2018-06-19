from __future__ import print_function, absolute_import

import numba.unittest_support as unittest
from numba import hsa


class TestLinkage(unittest.TestCase):
    def test_indirection(self):
        @hsa.jit(device=True)
        def base():
            pass

        @hsa.jit(device=True)
        def layer1():
            base()

        @hsa.jit(device=True)
        def layer2():
            layer1()
            base()

        @hsa.jit
        def kernel(a):
            layer2()

        kernel[1, 1](1)


if __name__ == '__main__':
    unittest.main()
