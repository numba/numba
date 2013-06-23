from numbapro import cuda, vectorize
from numba import *
import numpy as np
import unittest
from .support import addtest, main

@jit(f4(f4, f4, f4), device=True, inline=True, target='gpu')
def cu_device_fn(x, y, z):
    return x ** y / z

def cu_ufunc(x, y, z):
    return cu_device_fn(x, y, z)

@addtest
class TestCudaVectorizeDeviceCall(unittest.TestCase):

    def test_cuda_vectorize_device_call(self):

        ufunc = vectorize([f4(f4, f4, f4)], target='gpu')(cu_ufunc)


        N = 100

        X = np.array(np.random.sample(N), dtype=np.float32)
        Y = np.array(np.random.sample(N), dtype=np.float32)
        Z = np.array(np.random.sample(N), dtype=np.float32) + 0.1

        out = ufunc(X, Y, Z)

        gold = (X ** Y) / Z

        assert np.allclose(out, gold)

if __name__ == '__main__':
    main()