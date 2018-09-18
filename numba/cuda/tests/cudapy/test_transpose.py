import numpy as np
from numba import cuda
from numba.cuda.kernels.transpose import transpose
from numba.cuda.testing import unittest
from numba.testing.ddt import ddt, data, unpack
from numba.cuda.testing import skip_on_cudasim, SerialMixin


@skip_on_cudasim('Device Array API unsupported in the simulator')
@ddt
class Test(SerialMixin, unittest.TestCase):

    @data((5, 6, np.float64),
          (128, 128, np.complex128),
          (1025, 512, np.float64))
    @unpack
    def test_transpose(self, rows, cols, dtype):

        x = np.arange(rows * cols, dtype=dtype).reshape(cols, rows)
        y = np.zeros(rows * cols, dtype=dtype).reshape(rows, cols)
        dx = cuda.to_device(x)
        dy = cuda.cudadrv.devicearray.from_array_like(y)
        transpose(dx, dy)
        dy.copy_to_host(y)
        self.assertTrue(np.all(x.transpose() == y))


if __name__ == '__main__':
    unittest.main()
