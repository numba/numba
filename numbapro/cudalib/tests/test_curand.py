import numpy as np
import unittest
from .support import addtest

from numbapro import cuda

@addtest
class TestCURand(unittest.TestCase):
    def test_lib(self):
        from numbapro.cudalib.curand.binding import libcurand
        curand = libcurand()
        print('curand version %d' % libcurand().version)
        self.assertNotEqual(libcurand().version, 0)


class TestCURandPseudo(unittest.TestCase):
    def setUp(self):
        from numbapro.cudalib.curand.binding import (Generator,
                                                     CURAND_RNG_PSEUDO_DEFAULT)

        self.N = 10
        self.ary32 = np.zeros(self.N, dtype=np.float32)
        self.ary64 = np.zeros(self.N, dtype=np.float64)

        self.stream = cuda.stream()
        self.devary32 = cuda.to_device(self.ary32, stream=self.stream)
        self.devary64 = cuda.to_device(self.ary64, stream=self.stream)

        self.rndgen = Generator(CURAND_RNG_PSEUDO_DEFAULT)
        self.rndgen.set_stream(self.stream)
        self.rndgen.set_offset(123)
        self.rndgen.set_pseudo_random_generator_seed(1234)

    def tearDown(self):
        self.devary32.copy_to_host(self.ary32, stream=self.stream)
        self.devary64.copy_to_host(self.ary64, stream=self.stream)

        self.stream.synchronize()

        self.assertTrue(any(self.ary32 != 0))
        self.assertTrue(any(self.ary64 != 0))

        del self.N
        del self.ary32
        del self.ary64
        del self.stream
        del self.devary32
        del self.devary64

    def test_uniform(self):
        self.rndgen.generate_uniform(self.devary32, self.N)
        self.rndgen.generate_uniform(self.devary64, self.N)


    def test_normal(self):
        self.rndgen.generate_normal(self.devary32, self.N, 0, 1)
        self.rndgen.generate_normal(self.devary64, self.N, 0, 1)


    def test_log_normal(self):
        self.rndgen.generate_log_normal(self.devary32, self.N, 0, 1)
        self.rndgen.generate_log_normal(self.devary64, self.N, 0, 1)


class TestCURandPoisson(unittest.TestCase):
    def setUp(self):
        from numbapro.cudalib.curand.binding import (Generator,
                                                     CURAND_RNG_PSEUDO_DEFAULT)

        self.N = 10
        self.ary32 = np.zeros(self.N, dtype=np.uint32)

        self.stream = cuda.stream()
        self.devary32 = cuda.to_device(self.ary32, stream=self.stream)

        self.rndgen = Generator(CURAND_RNG_PSEUDO_DEFAULT)
        self.rndgen.set_stream(self.stream)
        self.rndgen.set_offset(123)
        self.rndgen.set_pseudo_random_generator_seed(1234)

    def tearDown(self):
        self.devary32.copy_to_host(self.ary32, stream=self.stream)

        self.stream.synchronize()

        self.assertTrue(any(self.ary32 != 0))

        del self.N
        del self.ary32
        del self.stream
        del self.devary32

    def test_poisson(self):
        self.rndgen.generate_poisson(self.devary32, self.N, 1)



class TestCURandQuasi(unittest.TestCase):
    def test_generate(self):
        from numbapro.cudalib.curand.binding import (Generator,
                                                     CURAND_RNG_QUASI_SOBOL64,
                                                     CURAND_RNG_QUASI_DEFAULT)
        N = 10
        stream = cuda.stream()

        ary32 = np.zeros(N, dtype=np.uint32)
        devary32 = cuda.to_device(ary32, stream=stream)

        rndgen = Generator(CURAND_RNG_QUASI_DEFAULT)
        rndgen.set_stream(stream)
        rndgen.set_offset(123)
        rndgen.set_quasi_random_generator_dimensions(1)
        rndgen.generate(devary32, N)

        devary32.copy_to_host(ary32, stream=stream)
        stream.synchronize()

        self.assertTrue(any(ary32 != 0))


        ary64 = np.zeros(N, dtype=np.uint64)
        devary64 = cuda.to_device(ary64, stream=stream)

        rndgen = Generator(CURAND_RNG_QUASI_SOBOL64)
        rndgen.set_stream(stream)
        rndgen.set_offset(123)
        rndgen.set_quasi_random_generator_dimensions(1)
        rndgen.generate(devary64, N)

        devary64.copy_to_host(ary64, stream=stream)
        stream.synchronize()

        self.assertTrue(any(ary64 != 0))


class TestCURandAPI(unittest.TestCase):
    def test_pseudo(self):
        from numbapro.cudalib import curand
        prng = curand.PRNG()
        prng.seed = 0xbeef
        N = 10
        ary = np.zeros(N, dtype=np.float32)
        prng.uniform(ary, N)
        self.assertTrue(any(ary != 0))

        iary = np.zeros(N, dtype=np.uint32)
        prng.poisson(iary, N)
        self.assertTrue(any(iary != 0))

    def test_quasi(self):
        from numbapro.cudalib import curand
        qrng = curand.QRNG()
        qrng.ndim = 2
        N = 10
        ary = np.zeros(N, dtype=np.uint32)
        qrng.generate(ary, N)
        self.assertTrue(any(ary != 0))


class TestTopLevel(unittest.TestCase):
    def test_uniform(self):
        from numbapro.cudalib import curand
        A = curand.uniform(10)
        B = curand.uniform(10)
        self.assertTrue(np.mean(abs((A - B) / B)) > .10)

    def test_normal(self):
        from numbapro.cudalib import curand
        A = curand.normal(0, 1, 10)
        B = curand.normal(0, 1, 10)
        self.assertTrue(np.mean(abs((A - B) / B)) > .10)

    def test_lognormal(self):
        from numbapro.cudalib import curand
        A = curand.lognormal(0, 1, 10)
        B = curand.lognormal(0, 1, 10)
        self.assertTrue(np.mean(abs((A - B) / B)) > .10)

    def test_poisson(self):
        from numbapro.cudalib import curand
        A = curand.poisson(10, 10)
        B = curand.poisson(10, 10)
        self.assertTrue(np.mean(abs((A - B) / B)) > .10)

    def test_quasi(self):
        from numbapro.cudalib import curand
        A = curand.quasi(10, nd=1, bits=32)
        B = curand.quasi(10, nd=1, bits=32)
        self.assertTrue(np.mean(abs((A - B) / B)) > .10)

        A = curand.quasi(10, nd=1, bits=64)
        B = curand.quasi(10, nd=1, bits=64)
        self.assertTrue(np.mean(abs((A - B) / B)) > .10)

        A = curand.quasi(10, nd=5, bits=32)
        B = curand.quasi(10, nd=5, bits=32)
        self.assertTrue(np.mean(abs((A - B) / B)) > .10)

        A = curand.quasi(10, nd=5, bits=64)
        B = curand.quasi(10, nd=5, bits=64)
        self.assertTrue(np.mean(abs((A - B) / B)) > .10)


if __name__ == '__main__':
    unittest.main()

