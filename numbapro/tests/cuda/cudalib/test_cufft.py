import numpy as np
import unittest

from numbapro import cuda
from numbapro.cudalib.cufft.binding import Plan, CUFFT_C2C, CUFFT_FORWARD, CUFFT_INVERSE

class TestCuFFTLib(unittest.TestCase):
    def test_lib(self):
        from numbapro.cudalib.cufft.binding import libcufft
        cufft = libcufft()
        print('cufft version %d' % libcufft().version)
        self.assertNotEqual(libcufft().version, 0)

class TestCuFFTPlan(unittest.TestCase):
    def test_plan1d(self):
        n = 10
        data = np.arange(n, dtype=np.complex64)
        orig = data.copy()
        d_data = cuda.to_device(data)
        fftplan = Plan.one(CUFFT_C2C, n)
        fftplan.exe(d_data, d_data, CUFFT_FORWARD)
        fftplan.exe(d_data, d_data, CUFFT_INVERSE)
        d_data.to_host()
        result = data / n
        self.assertTrue(np.allclose(orig, result.real))

if __name__ == '__main__':
    unittest.main()