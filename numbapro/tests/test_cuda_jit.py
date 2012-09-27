import numpy as np
from numba import *
from numbapro import cuda

@cuda.jit(argtypes=[f4[:], f4[:]])
def array_copy(src, dst):
    tid = cuda.threadIdx.x
    blkid = cuda.blockIdx.x
    blkdim = cuda.blockDim.x

    i = tid + blkid * blkdim

    dst[i] = src[i]

def main():
    N = 1024 * 2
    src = np.arange(N, dtype=np.float32)
    dst = np.empty_like(src)

    src_copy = src.copy()
    dst_copy = dst.copy()

    array_copy.configure((2,), (1024,))
    array_copy(src, dst)

    assert (src == dst).all()

if __name__ == '__main__':
    for i in range(100):
        main()
