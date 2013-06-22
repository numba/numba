import numpy as np
import unittest
from .support import addtest

from numbapro import cuda

@addtest
class TestCuFFTLib(unittest.TestCase):
    def test_lib(self):
        from numbapro.cudalib.cufft.binding import libcufft
        cufft = libcufft()
        print('cufft version %d' % libcufft().version)
        self.assertNotEqual(libcufft().version, 0)

class TestCuFFTPlan(unittest.TestCase):
    def test_plan1d(self):
        from numbapro.cudalib.cufft.binding import Plan, CUFFT_C2C
        n = 10
        data = np.arange(n, dtype=np.complex64)
        orig = data.copy()
        d_data = cuda.to_device(data)
        fftplan = Plan.one(CUFFT_C2C, n)
        fftplan.forward(d_data, d_data)
        fftplan.inverse(d_data, d_data)
        d_data.copy_to_host(data)
        result = data / n
        self.assertTrue(np.allclose(orig, result.real))

    def test_plan2d(self):
        from numbapro.cudalib.cufft.binding import Plan, CUFFT_C2C
        n = 2**4
        data = np.arange(n, dtype=np.complex64).reshape(2, n//2)
        orig = data.copy()
        d_data = cuda.to_device(data)
        fftplan = Plan.two(CUFFT_C2C, *data.shape)
        fftplan.forward(d_data, d_data)
        fftplan.inverse(d_data, d_data)
        d_data.copy_to_host(data)
        result = data / n
        self.assertTrue(np.allclose(orig, result.real))

    def test_plan3d(self):
        from numbapro.cudalib.cufft.binding import Plan, CUFFT_C2C
        n = 32
        data = np.arange(n, dtype=np.complex64).reshape(2, 2, 8)
        fft = np.empty_like(data)
        orig = data.copy()
        d_data = cuda.to_device(data)
        fftplan = Plan.three(CUFFT_C2C, *data.shape)
        fftplan.forward(d_data, d_data)
        fftplan.inverse(d_data, d_data)
        d_data.copy_to_host(data)
        result = data / n
        self.assertTrue(np.allclose(orig, result.real))


    def test_against_fft_1d(self):
        from numbapro.cudalib.cufft.binding import Plan, CUFFT_R2C
        N = 128
        x = np.asarray(np.arange(N), dtype=np.float32)
        xf = np.fft.fft(x)
        d_x_gpu = cuda.to_device(x)
        xf_gpu = np.zeros(N//2+1, np.complex64)
        d_xf_gpu = cuda.to_device(xf_gpu)
        plan = Plan.many(x.shape, CUFFT_R2C)
        plan.forward(d_x_gpu, d_xf_gpu)
        d_xf_gpu.copy_to_host(xf)
        self.assertTrue( np.allclose(xf[0:N/2+1], xf_gpu, atol=1e-6) )

    def test_against_fft_2d(self):
        from numbapro.cudalib.cufft.binding import Plan, CUFFT_R2C
        rank = 2
        rowsize = 128
        N = rowsize * rank
        x = np.arange(N, dtype=np.float32).reshape(rank, rowsize)
        xf = np.fft.fft2(x)
        d_x_gpu = cuda.to_device(x)
        xf_gpu = np.zeros(shape=(rank, rowsize//2 + 1), dtype=np.complex64)
        d_xf_gpu = cuda.to_device(xf_gpu)
        plan = Plan.many(x.shape, CUFFT_R2C)
        plan.forward(d_x_gpu, d_xf_gpu)
        d_xf_gpu.copy_to_host(xf)
        self.assertTrue(np.allclose(xf[:, 0:rowsize//2+1], xf_gpu, atol=1e-6))

    def test_against_fft_3d(self):
        from numbapro.cudalib.cufft.binding import Plan, CUFFT_R2C
        depth = 2
        colsize = 2
        rowsize = 64
        N = depth * colsize * rowsize
        x = np.arange(N, dtype=np.float32).reshape(depth, colsize, rowsize)
        xf = np.fft.fftn(x)
        d_x_gpu = cuda.to_device(x)
        xf_gpu = np.zeros(shape=(depth, colsize, rowsize//2 + 1), dtype=np.complex64)
        d_xf_gpu = cuda.to_device(xf_gpu)
        plan = Plan.many(x.shape, CUFFT_R2C)
        plan.forward(d_x_gpu, d_xf_gpu)
        d_xf_gpu.copy_to_host(xf)
        self.assertTrue(np.allclose(xf[:, :, 0:rowsize//2+1], xf_gpu, atol=1e-6))


class TestCuFFTAPI(unittest.TestCase):
    def test_fft_1d_single(self):
        from numbapro.cudalib.cufft import FFTPlan, fft
        N = 32
        x = np.asarray(np.arange(N), dtype=np.float32)
        xf = np.fft.fft(x)

        xf_gpu = np.empty(shape=N//2 + 1, dtype=np.complex64)
        fft(x, xf_gpu)

        self.assertTrue( np.allclose(xf[0:N/2+1], xf_gpu, atol=1e-6) )

    def test_fft_1d_double(self):
        from numbapro.cudalib.cufft import FFTPlan, fft
        N = 32
        x = np.asarray(np.arange(N), dtype=np.float64)
        xf = np.fft.fft(x)

        xf_gpu = np.zeros(shape=N//2 + 1, dtype=np.complex128)
        fft(x, xf_gpu)

        self.assertTrue( np.allclose(xf[0:N/2+1], xf_gpu, atol=1e-6) )

    def test_fft_2d_single(self):
        from numbapro.cudalib.cufft import FFTPlan, fft
        N2 = 2
        N1 = 32
        N = N1 * N2
        x = np.asarray(np.arange(N), dtype=np.float32).reshape(N2, N1)
        xf = np.fft.fft2(x)

        xf_gpu = np.empty(shape=(N2, N1//2 + 1), dtype=np.complex64)
        fft(x, xf_gpu)

        self.assertTrue( np.allclose(xf[:, 0:N1/2+1], xf_gpu, atol=1e-6) )


    def test_fft_2d_double(self):
        from numbapro.cudalib.cufft import FFTPlan, fft
        N2 = 2
        N1 = 32
        N = N1 * N2
        x = np.asarray(np.arange(N), dtype=np.float64).reshape(N2, N1)
        xf = np.fft.fft2(x)

        xf_gpu = np.empty(shape=(N2, N1//2 + 1), dtype=np.complex128)
        fft(x, xf_gpu)

        self.assertTrue( np.allclose(xf[:, 0:N1/2+1], xf_gpu, atol=1e-6) )


    def test_fft_3d_single(self):
        from numbapro.cudalib.cufft import FFTPlan, fft
        N3 = 2
        N2 = 2
        N1 = 32
        N = N1 * N2 * N3
        x = np.asarray(np.arange(N), dtype=np.float32).reshape(N3, N2, N1)
        xf = np.fft.fftn(x)

        xf_gpu = np.empty(shape=(N3, N2, N1//2 + 1), dtype=np.complex64)
        fft(x, xf_gpu)

        self.assertTrue( np.allclose(xf[:, :, 0:N1/2+1], xf_gpu, atol=1e-6) )

    def test_fft_3d_double(self):
        from numbapro.cudalib.cufft import FFTPlan, fft
        N3 = 2
        N2 = 2
        N1 = 32
        N = N1 * N2 * N3
        x = np.asarray(np.arange(N), dtype=np.float64).reshape(N3, N2, N1)
        xf = np.fft.fftn(x)

        xf_gpu = np.empty(shape=(N3, N2, N1//2 + 1), dtype=np.complex128)
        fft(x, xf_gpu)

        self.assertTrue( np.allclose(xf[:, :, 0:N1/2+1], xf_gpu, atol=1e-6) )

    def test_fft_1d_roundtrip_single(self):
        from numbapro.cudalib.cufft import FFTPlan, fft, ifft
        N = 32
        x = np.asarray(np.arange(N), dtype=np.float32)
        x0 = x.copy()
        xf_gpu = np.empty(shape=N//2 + 1, dtype=np.complex64)
        fft(x, xf_gpu)
        ifft(xf_gpu, x)
        self.assertTrue( np.allclose(x / N, x0, atol=1e-6) )

    def test_fft_1d_roundtrip_double(self):
        from numbapro.cudalib.cufft import FFTPlan, fft, ifft
        N = 32
        x = np.asarray(np.arange(N), dtype=np.float64)
        x0 = x.copy()
        xf_gpu = np.empty(shape=N//2 + 1, dtype=np.complex128)
        fft(x, xf_gpu)
        ifft(xf_gpu, x)
        self.assertTrue( np.allclose(x / N, x0, atol=1e-6) )


    def test_fft_2d_roundtrip_single(self):
        from numbapro.cudalib.cufft import FFTPlan, fft, ifft
        N2 = 2
        N1 = 32
        N = N2 * N1
        x = np.asarray(np.arange(N), dtype=np.float32).reshape(N2, N1)
        x0 = x.copy()
        xf_gpu = np.empty(shape=(N2, N1//2 + 1), dtype=np.complex64)
        fft(x, xf_gpu)
        ifft(xf_gpu, x)
        self.assertTrue( np.allclose(x / N, x0, atol=1e-6) )

    def test_fft_3d_roundtrip_single(self):
        from numbapro.cudalib.cufft import FFTPlan, fft, ifft
        N3 = 2
        N2 = 2
        N1 = 32
        N = N3 * N2 * N1
        x = np.asarray(np.arange(N), dtype=np.float32).reshape(N3, N2, N1)
        x0 = x.copy()
        xf_gpu = np.empty(shape=(N3, N2, N1//2 + 1), dtype=np.complex64)
        fft(x, xf_gpu)
        ifft(xf_gpu, x)
        self.assertTrue( np.allclose(x / N, x0, atol=1e-6) )

    def test_fft_inplace_1d_single(self):
        from numbapro.cudalib.cufft import FFTPlan, fft_inplace
        N = 32
        x = np.asarray(np.arange(N), dtype=np.complex64)
        xf = np.fft.fft(x)

        fft_inplace(x)

        self.assertTrue( np.allclose(xf, x, atol=1e-6) )

    def test_fft_inplace_1d_double(self):
        from numbapro.cudalib.cufft import FFTPlan, fft_inplace
        N = 32
        x = np.asarray(np.arange(N), dtype=np.complex128)
        xf = np.fft.fft(x)

        fft_inplace(x)

        self.assertTrue( np.allclose(xf, x, atol=1e-6) )

    def test_fft_inplace_2d_single(self):
        from numbapro.cudalib.cufft import FFTPlan, fft_inplace
        N1 = 32
        N2 = 2
        N = N1 * N2
        x = np.asarray(np.arange(N), dtype=np.complex64).reshape(N2, N1)
        xf = np.fft.fft2(x)

        fft_inplace(x)

        self.assertTrue( np.allclose(xf, x, atol=1e-6) )

    def test_fft_inplace_2d_double(self):
        from numbapro.cudalib.cufft import FFTPlan, fft_inplace
        N1 = 32
        N2 = 2
        N = N1 * N2
        x = np.asarray(np.arange(N), dtype=np.complex128).reshape(N2, N1)
        xf = np.fft.fft2(x)

        fft_inplace(x)

        self.assertTrue( np.allclose(xf, x, atol=1e-6) )

    def test_fft_1d_roundtrip_single(self):
        from numbapro.cudalib.cufft import FFTPlan, fft_inplace, ifft_inplace
        N = 32
        x = np.asarray(np.arange(N), dtype=np.complex64)
        x0 = x.copy()

        fft_inplace(x)
        ifft_inplace(x)
        
        self.assertTrue( np.allclose(x / N, x0, atol=1e-6) )

    def test_fft_1d_roundtrip_double(self):
        from numbapro.cudalib.cufft import FFTPlan, fft_inplace, ifft_inplace
        N = 32
        x = np.asarray(np.arange(N), dtype=np.complex128)
        x0 = x.copy()

        fft_inplace(x)
        ifft_inplace(x)
        
        self.assertTrue( np.allclose(x / N, x0, atol=1e-6) )


    def test_fft_2d_roundtrip_single(self):
        from numbapro.cudalib.cufft import FFTPlan, fft_inplace, ifft_inplace
        N2 = 2
        N1 = 32
        N = N1 * N2
        x = np.asarray(np.arange(N), dtype=np.complex64).reshape(N2, N1)
        x0 = x.copy()

        fft_inplace(x)
        ifft_inplace(x)
        
        self.assertTrue( np.allclose(x / N, x0, atol=1e-6) )

    def test_fft_3d_roundtrip_double(self):
        from numbapro.cudalib.cufft import FFTPlan, fft_inplace, ifft_inplace
        N3 = 2
        N2 = 2
        N1 = 8
        N = N3 * N2 * N1
        x = np.asarray(np.arange(N), dtype=np.complex128).reshape(N3, N2, N1)
        x0 = x.copy()

        fft_inplace(x)
        ifft_inplace(x)
        
        self.assertTrue( np.allclose(x / N, x0, atol=1e-6) )







if __name__ == '__main__':
    unittest.main()

