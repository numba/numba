import numpy as np
from numba import cuda
from numba.cuda.testing import CUDATestCase, skip_on_cudasim
from numba.tests.support import override_config
from numba.core.errors import NumbaPerformanceWarning
import warnings


def numba_dist_cuda(a, b, dist):
    len = a.shape[0]
    for i in range(len):
        dist[i] = a[i] * b[i]


def numba_dist_cuda2(a, b, dist):
    len = a.shape[0]
    len2 = a.shape[1]
    for i in range(len):
        for j in range(len2):
            dist[i, j] = a[i, j] * b[i, j]


@skip_on_cudasim('Large data set causes slow execution in the simulator')
class TestCUDAWarnings(CUDATestCase):
    def test_inefficient_kernel(self):
        a = np.random.rand(1024 * 1024 * 32).astype('float32')
        b = np.random.rand(1024 * 1024 * 32).astype('float32')
        dist = np.zeros(a.shape[0]).astype('float32')

        sig = 'void(float32[:], float32[:], float32[:])'
        with override_config('CUDA_LOW_OCCUPANCY_WARNINGS', 1):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always', NumbaPerformanceWarning)
                cuda_func = cuda.jit(sig)(numba_dist_cuda)
                cuda_func[1,1](a, b, dist)
                self.assertEqual(w[0].category, NumbaPerformanceWarning)
                self.assertIn('Grid size', str(w[0].message))
                self.assertIn('2 * SM count', str(w[0].message))

    def test_efficient_kernel(self):
        a = np.random.rand(1024 * 1024 * 128).astype('float32').\
            reshape((1024 * 1024, 128))
        b = np.random.rand(1024 * 1024 * 128).astype('float32').\
            reshape((1024 * 1024, 128))
        dist = np.zeros_like(a)

        sig = 'void(float32[:, :], float32[:, :], float32[:, :])'

        with override_config('CUDA_LOW_OCCUPANCY_WARNINGS', 1):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always', NumbaPerformanceWarning)
                cuda_func = cuda.jit(sig)(numba_dist_cuda2)
                cuda_func[256,256](a, b, dist)
                self.assertEqual(len(w), 0)
