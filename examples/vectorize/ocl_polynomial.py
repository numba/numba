#! /usr/bin/env python
from __future__ import print_function

import sys
from timeit import default_timer as time

import numpy as np

from numba import vectorize, ocl

import polynomial as poly


def main():
    cl_discriminant = vectorize(['f4(f4, f4, f4)', 'f8(f8, f8, f8)'],
                                target='ocl')(poly.discriminant)

    N = 1e+8 // 2
    N = int(N)

    print('Data size', N)

    A, B, C = poly.generate_input(N, dtype=np.float32)
    D = np.empty(A.shape, dtype=A.dtype)

    stream = ocl.stream()

    print('== One')

    ts = time()

    with stream.auto_synchronize():
        dA = ocl.to_device(A, stream)
        dB = ocl.to_device(B, stream)
        dC = ocl.to_device(C, stream)
        dD = ocl.to_device(D, stream, copy=False)
        cl_discriminant(dA, dB, dC, out=dD, stream=stream)
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
            dA = ocl.to_device(a, stream)
            dB = ocl.to_device(b, stream)
            dC = ocl.to_device(c, stream)
            dD = ocl.to_device(d, stream, copy=False)
            cl_discriminant(dA, dB, dC, out=dD, stream=stream)
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
