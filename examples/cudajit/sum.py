import numpy as np
from numba import *
from numbapro import cuda


@cuda.jit(argtypes=[f4[:], f4[:], f4[:]])
def cuda_sum(a, b, c):
    tid = cuda.threadIdx.x
    blkid = cuda.blockIdx.x
    blkdim = cuda.blockDim.x
    i = tid + blkid * blkdim
    c[i] = a[i] + b[i]

griddim = 10, 1
blockdim = 32, 1, 1

cuda_sum_configured = cuda_sum.configure(griddim, blockdim)
a = np.array(np.random.random(320), dtype=np.float32)
b = np.array(np.random.random(320), dtype=np.float32)
c = np.empty_like(a)
cuda_sum_configured(a, b, c)

assert (a + b == c).all()
#print c
