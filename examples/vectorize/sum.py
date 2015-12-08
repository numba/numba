#! /usr/bin/env python

'''
Example vectorize usage.
'''

import numpy as np
from numba import *
from timeit import default_timer as time
import math
import sys

def check_answer(ans, A, B, C):
    for d, a, b in zip(ans, A, B):
        gold = sum(a, b)
        assert np.allclose(d, gold), (d, gold)

def sum(a, b):
    return a + b

def main():
    targets = ['cpu', 'stream', 'parallel']
    
    # run just one target if is specified in the argument
    for t in targets:
        if t in sys.argv[1:]:
            targets = [t]
            break

    for target in targets:
        print('== Target', target)
        vect_sum = vectorize([f4(f4, f4), f8(f8, f8)],
                             target=target)(sum)

        A = np.fromfile('inputA.dat', dtype=np.float32)
        B = np.fromfile('inputB.dat', dtype=np.float32)
        assert A.shape == B.shape
        assert A.dtype ==  B.dtype
        assert len(A.shape) == 1
        N = A.shape[0]
        D = np.empty(A.shape, dtype=A.dtype)

        print('Data size', N)

        ts = time()
        D = vect_sum(A, B)
        te = time()

        total_time = (te - ts)

        print('Execution time %.4f' % total_time)
        print('Throughput %.4f' % (N / total_time))



        if '-verify' in sys.argv[1:]:
            check_answer(D, A, B, C)


if __name__ == '__main__':
    main()
