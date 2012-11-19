'''
Example vectorize usage.
'''

import numpy as np
from numba import *
from numbapro import vectorize
from timeit import default_timer as time
import math
from itertools import izip
import sys

def generate_input(n, dtype):
    A = np.array(np.random.sample(n), dtype=dtype)
    B = np.array(np.random.sample(n), dtype=dtype)
    return A, B

def check_answer(ans, A, B, C):
    for d, a, b in izip(ans, A, B):
        gold = sum(a, b)
        assert np.allclose(d, gold), (d, gold)

def sum(a, b):
    return a + b


def main():

    N = 1e+7
    print 'Data size', N

    targets = ['cpu', 'stream', 'parallel']
    
    # run just one target if is specified in the argument
    for t in targets:
        if t in sys.argv[1:]:
            targets = [t]
            break

    for target in targets:
        print '== Target', target
        vect_sum = vectorize([f4(f4, f4), f8(f8, f8)],
                             target=target)(sum)

        A, B = generate_input(N, dtype=np.float32)
        D = np.empty(A.shape, dtype=A.dtype)

        ts = time()
        D = vect_sum(A, B)
        te = time()

        total_time = (te - ts)

        print 'Execution time %.4f' % total_time
        print 'Throughput %.4f' % (N / total_time)



        if '-verify' in sys.argv[1:]:
            check_answer(D, A, B, C)


if __name__ == '__main__':
    main()
