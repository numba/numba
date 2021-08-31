from numba import cuda, float64, uint32, uint64
from numba.cuda.testing import unittest, CUDATestCase


class TestCudaMandel(CUDATestCase):
    def test_mandel(self):
        """Just make sure we can compile this
        """
        def mandel(tid, min_x, max_x, min_y, max_y, width, height, iters):
            pixel_size_x = (max_x - min_x) / width
            pixel_size_y = (max_y - min_y) / height

            x = tid % width
            y = tid / width

            real = min_x + x * pixel_size_x
            imag = min_y + y * pixel_size_y

            c = complex(real, imag)
            z = 0.0j

            for i in range(iters):
                z = z * z + c
                if (z.real * z.real + z.imag * z.imag) >= 4:
                    return i
            return iters

        args = (uint32, float64, float64, float64, float64, uint32, uint32,
                uint32)

        ptx, resty = cuda.compile_ptx_for_current_device(mandel, args,
                                                         device=True)
        # Check return type
        self.assertEqual(resty, uint64)
        # Check a device function was emitted
        self.assertIn('.visible .func', ptx)


if __name__ == '__main__':
    unittest.main()
