from __future__ import print_function, absolute_import
from numba import unittest_support as unittest
from numba.ocl.testing import OCLTestCase
from numba.ocl import compiler
from numba import ocl
from numba import types
import numpy as np


class TestCompiler(OCLTestCase):
    def test_single_element_assign(self):
        def pyfunc(x):
            x[0] = 1234

        argtys = [types.Array(types.int32, 1, 'C')]
        kern = compiler.compile_kernel(pyfunc, argtys)

        data = np.zeros(1, dtype='int32')

        dev_data = ocl.to_device(data)
        kern[1, 1](dev_data)
        dev_data.copy_to_host(data)

        self.assertEqual(data[0], 1234)

    def test_get_global_id(self):
        def pyfunc(x):
            i = ocl.get_global_id(0)
            x[i] = i + 1

        argtys = [types.Array(types.int32, 1, 'C')]
        kern = compiler.compile_kernel(pyfunc, argtys)

        data = np.zeros(10, dtype='int32')

        dev_data = ocl.to_device(data)
        kern[data.size, 1](dev_data)
        dev_data.copy_to_host(data)

        self.assertTrue(np.all(data == 1 + np.arange(data.size)))

    def test_get_local_id(self):
        def pyfunc(x):
            i = ocl.get_local_id(0)
            x[i] = i + 1

        argtys = [types.Array(types.int32, 1, 'C')]
        kern = compiler.compile_kernel(pyfunc, argtys)

        data = np.zeros(10, dtype='int32')

        dev_data = ocl.to_device(data)
        kern[1, data.size](dev_data)
        dev_data.copy_to_host(data)

        self.assertTrue(np.all(data == 1 + np.arange(data.size)))

    def test_get_global_size(self):
        def pyfunc(x):
            i = ocl.get_global_id(0)
            n = ocl.get_global_size(0)
            m = ocl.get_local_size(0)
            x[i] = n * m

        argtys = [types.Array(types.int32, 1, 'C')]
        kern = compiler.compile_kernel(pyfunc, argtys)

        data = np.zeros(32, dtype='int32')

        dev_data = ocl.to_device(data)
        kern[1, data.size](dev_data)
        dev_data.copy_to_host(data)

        self.assertTrue(np.all(data == data.size ** 2))

if __name__ == '__main__':
    unittest.main()

# ; Function Attrs: nounwind readnone
# declare cc75 i32 @_Z13get_global_idj(i32) #1