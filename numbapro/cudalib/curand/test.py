import numpy as np
import unittest

from numbapro import cuda
import api

class TestCURand(unittest.TestCase):
    def test_lib(self):
        curand = api.libcurand()
        print('curand version %d' % api.libcurand().version)
        self.assertNotEqual(api.libcurand().version, 0)


class TestCURandPseudo(unittest.TestCase):
    def setUp(self):
        self.N = 10
        self.ary32 = np.zeros(self.N, dtype=np.float32)
        self.ary64 = np.zeros(self.N, dtype=np.float64)

        self.stream = cuda.stream()
        self.devary32 = cuda.to_device(self.ary32, stream=self.stream)
        self.devary64 = cuda.to_device(self.ary64, stream=self.stream)

        self.rndgen = api.Generator(api.CURAND_RNG_PSEUDO_DEFAULT)
        self.rndgen.set_stream(self.stream)
        self.rndgen.set_offset(123)
        self.rndgen.set_pseudo_random_generator_seed(1234)

    def tearDown(self):
        self.devary32.to_host(stream=self.stream)
        self.devary64.to_host(stream=self.stream)

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

    def test_poisson(self):
        self.rndgen.generate_poisson(self.devary32, self.N, 1)
        self.rndgen.generate_poisson(self.devary64, self.N, 1)


class TestCURandQuasi(unittest.TestCase):
    def test_generate(self):
        N = 10
        stream = cuda.stream()

        ary32 = np.zeros(N, dtype=np.uint32)
        devary32 = cuda.to_device(ary32, stream=stream)

        rndgen = api.Generator(api.CURAND_RNG_QUASI_DEFAULT)
        rndgen.set_stream(stream)
        rndgen.set_offset(123)
        rndgen.set_quasi_random_generator_dimensions(1)
        rndgen.generate(devary32, N)

        devary32.to_host(stream=stream)
        stream.synchronize()

        self.assertTrue(any(ary32 != 0))


        ary64 = np.zeros(N, dtype=np.uint64)
        devary64 = cuda.to_device(ary64, stream=stream)

        rndgen = api.Generator(api.CURAND_RNG_QUASI_SOBOL64)
        rndgen.set_stream(stream)
        rndgen.set_offset(123)
        rndgen.set_quasi_random_generator_dimensions(1)
        rndgen.generate(devary64, N)

        devary64.to_host(stream=stream)
        stream.synchronize()

        self.assertTrue(any(ary64 != 0))




if __name__ == '__main__':
    unittest.main()
