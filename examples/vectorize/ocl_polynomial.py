#! /usr/bin/env python
from __future__ import print_function

import sys
from timeit import default_timer as time

import numpy as np

from numba import vectorize, cuda

import polynomial as poly


def main():
    cu_discriminant = vectorize(['f4(f4, f4, f4)', 'f8(f8, f8, f8)'],
                                target='cuda')(poly.discriminant)

    N = 1e+8 // 2

    print('Data size', N)

    A, B, C = poly.generate_input(N, dtype=np.float32)
    D = np.empty(A.shape, dtype=A.dtype)

    stream = cuda.stream()

    print('== One')

    ts = time()

    with stream.auto_synchronize():
        dA = cuda.to_device(A, stream)
        dB = cuda.to_device(B, stream)
        dC = cuda.to_device(C, stream)
        dD = cuda.to_device(D, stream, copy=False)
        cu_discriminant(dA, dB, dC, out=dD, stream=stream)
        dD.to_host(stream)

    te = time()


    total_time = (te - ts)

    print('Execution time %.4f' % total_time)
    print('Throughput %.2f' % (N / total_time))

    print('== Chunked')

    chunksize = 1e+7
    chunkcount = N // chunksize

    print('Chunk size', chunksize)

    sA = np.split(A, chunkcount)
    sB = np.split(B, chunkcount)
    sC = np.split(C, chunkcount)
    sD = np.split(D, chunkcount)

    device_ptrs = []

    ts = time()

    with stream.auto_synchronize():
        for a, b, c, d in zip(sA, sB, sC, sD):
            dA = cuda.to_device(a, stream)
            dB = cuda.to_device(b, stream)
            dC = cuda.to_device(c, stream)
            dD = cuda.to_device(d, stream, copy=False)
            cu_discriminant(dA, dB, dC, out=dD, stream=stream)
            dD.to_host(stream)
            device_ptrs.extend([dA, dB, dC, dD])

    te = time()

    total_time = (te - ts)

    print('Execution time %.4f' % total_time)
    print('Throughput %.2f' % (N / total_time))


    if '-verify' in sys.argv[1:]:
        poly.check_answer(D, A, B, C)


if __name__ == '__main__':
    main()
