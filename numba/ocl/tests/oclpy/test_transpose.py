import numpy as np
from numba import ocl
from numba.ocl.kernels.transpose import transpose
from numba.ocl.testing import unittest
from numba.testing.ddt import ddt, data, unpack
from numba.ocl.testing import skip_on_oclsim

@ddt
class Test(unittest.TestCase):
        
    @data((5, 6, np.float64),
          (128, 128, np.complex128),
          (1025, 512, np.float64))
    @unpack
    def test_transpose(self, rows, cols, dtype):

        x = np.arange(rows * cols, dtype=dtype).reshape(cols, rows)
        y = np.zeros(rows * cols, dtype=dtype).reshape(rows, cols)
        dx = ocl.to_device(x)
        dy = ocl.ocldrv.devicearray.from_array_like(y)
        transpose(dx, dy)
        dy.copy_to_host(y)
        self.assertTrue(np.all(x.transpose() == y))


if __name__ == '__main__':
    unittest.main()
