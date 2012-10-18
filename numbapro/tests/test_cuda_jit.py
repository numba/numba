import numpy as np
from numba import *
from numbapro import cuda


def array_copy(src, dst, n):
    tid = cuda.threadIdx.x
    blkid = cuda.blockIdx.x
    blkdim = cuda.blockDim.x

    i = tid + blkid * blkdim

    if i >= n:
        return

    dst[i] = src[i]


def array_scale(src, dst, scale, n):
    tid = cuda.threadIdx.x
    blkid = cuda.blockIdx.x
    blkdim = cuda.blockDim.x

    i = tid + blkid * blkdim

    if i >= n:
        return

    dst[i] = src[i] * scale

def test_array_copy():
    N = 2 * 333
    src = np.arange(N, dtype=np.float32)
    dst = np.empty_like(src)

    prototype = jit2(argtypes=[f4[:], f4[:], i4], target='gpu')

    cudafunc = prototype(array_copy)
    cudafunc[(2,), (333,)](src, dst, N)

    assert (src == dst).all()

def test_array_copy_autojit():
    N = 2 * 333
    src = np.arange(N, dtype=np.float32)
    dst = np.empty_like(src)

    cudafunc = autojit(target='gpu')(array_copy)
    cudafunc[(2,), (333,)](src, dst, N)

    assert (src == dst).all()

def test_array_scale():
    N = 333 * 3
    scale = 3.14
    src = np.arange(N, dtype=np.float32)
    dst = np.empty_like(src)

    prototype = jit2(argtypes=[f4[:], f4[:], f4, i4], target='gpu')
    cudafunc = prototype(array_scale)
    cudafunc_configured = cudafunc.configure((3,), (333,))
    cudafunc_configured(src, dst, scale, N)

    assert (src * scale == dst).all()


def main():
    test_array_copy()
    test_array_copy_autojit()
    test_array_scale()

if __name__ == '__main__':
#    main()
    for i in range(100):
        #print cuda.cached
        main()
