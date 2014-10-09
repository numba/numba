from __future__ import print_function, division
import numpy as np
from numbapro import cuda
from numba.utils import benchmark

cuda_sum = cuda.Reduce(lambda a, b: a + b)

for p in range(8):
    n = 10 ** p
    print("n = {0} (10 ^ {1})".format(n, p))
    A = np.arange(n, dtype=np.float32) + 1

    ans_cpu = A.sum()
    ans_gpu = cuda_sum(A)
    assert np.allclose(ans_cpu, ans_gpu)
    print('CPU', benchmark(lambda: A.sum()))
    print('GPU', benchmark(lambda: cuda_sum(A)))
