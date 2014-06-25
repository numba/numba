from __future__ import print_function, absolute_import
from numba import unittest_support as unittest
from numba.ocl import compiler
from numba import ocl
from numba import types
import numpy as np


class TestCompiler(unittest.TestCase):
    def test_compiler_small(self):
        def pyfunc(x):
            x[0] = 1234

        argtys = [types.Array(types.int32, 1, 'C')]
        kern = compiler.compile_kernel(pyfunc, argtys)

        data = np.zeros(1, dtype='int32')

        dev_data = ocl.to_device(data)
        kern[1, 1](dev_data)
        dev_data.copy_to_host(data)
        print(data)


if __name__ == '__main__':
    unittest.main()

