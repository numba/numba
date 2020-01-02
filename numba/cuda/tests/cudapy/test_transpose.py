import numpy as np
from numba import cuda
from numba.cuda.kernels.transpose import transpose
from numba.cuda.testing import unittest
from numba.cuda.testing import skip_on_cudasim, SerialMixin


recordwith2darray = np.dtype([('i', np.int32),
                              ('j', np.float32, (3, 2))])


@skip_on_cudasim('Device Array API unsupported in the simulator')
class Test(SerialMixin, unittest.TestCase):

    def test_transpose(self):
        variants = ((5, 6, np.float64),
                    (128, 128, np.complex128),
                    (1025, 512, np.float64))

        for rows, cols, dtype in variants:
            with self.subTest(rows=rows, cols=cols, dtype=dtype):
                x = np.arange(rows * cols, dtype=dtype).reshape(cols, rows)
                y = np.zeros(rows * cols, dtype=dtype).reshape(rows, cols)
                dx = cuda.to_device(x)
                dy = cuda.cudadrv.devicearray.from_array_like(y)
                transpose(dx, dy)
                dy.copy_to_host(y)
                self.assertTrue(np.all(x.transpose() == y))

    def test_transpose_record(self):
        variants = ((2, 3), (16, 16), (16, 17), (17, 16), (14, 15), (15, 14),
                    (14, 14))
        for rows, cols in variants:
            with self.subTest(rows=rows, cols=cols):
                arr = np.recarray((rows, cols), dtype=recordwith2darray)
                for x in range(rows):
                    for y in range(cols):
                        arr[x, y].i = x**2 + y
                        j = np.arange(3 * 2, dtype=np.float32)
                        arr[x, y].j = j.reshape(3, 2) * x + y

                transposed = arr.T
                d_arr = cuda.to_device(arr)
                d_transposed = cuda.device_array_like(transposed)
                transpose(d_arr, d_transposed)
                host_transposed = d_transposed.copy_to_host()
                self.assertTrue(np.all(transposed == host_transposed))

    def test_transpose_bool(self):
        variants = ((2, 3), (16, 16), (16, 17), (17, 16), (14, 15), (15, 14),
                    (14, 14))
        for rows, cols in variants:
            with self.subTest(rows=rows, cols=cols):
                arr = np.random.randint(2, size=(rows, cols), dtype=np.bool_)
                transposed = arr.T

                d_arr = cuda.to_device(arr)
                d_transposed = cuda.device_array_like(transposed)
                transpose(d_arr, d_transposed)

                host_transposed = d_transposed.copy_to_host()
                self.assertTrue(np.all(transposed == host_transposed))


if __name__ == '__main__':
    unittest.main()
