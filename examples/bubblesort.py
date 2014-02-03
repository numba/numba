# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
from numba import *
import numpy as np
from timeit import default_timer as timer

def bubblesort(X, doprint):
    N = X.shape[0]
    for end in range(N, 1, -1):
        for i in range(end - 1):
            cur = X[i]
            if cur > X[i + 1]:
                tmp = X[i]
                X[i] = X[i + 1]
                X[i + 1] = tmp
#        if doprint:
#            print("Iteration: %d" % X)

# bubblesort_fast = autojit(bubblesort)
bubblesort_fast = jit(void(int64[::1], boolean))(bubblesort)

dtype = np.int64

def main():

    Xtest = np.array(list(reversed(range(8))), dtype=dtype)

    print('== Test Pure-Python ==')
    X0 = Xtest.copy()
    bubblesort(X0, True)

    print('== Test Numba == ')
    X1 = Xtest.copy()
    bubblesort_fast(X1, True)
#    return

    print(X0)
    print(X1)
    assert all(X0 == X1)

    REP = 10
    N = 1400

    Xorig = np.array(list(reversed(range(N))), dtype=dtype)

    t0 = timer()
    for t in range(REP):
        X0 = Xorig.copy()
        bubblesort(X0, False)
    tpython = (timer() - t0) / REP

    t1 = timer()
    for t in range(REP):
        X1 = Xorig.copy()
        bubblesort_fast(X1, False)
    tnumba = (timer() - t1) / REP

    assert all(X0 == X1)

    print('Python', tpython)
    print('Numba', tnumba)
    print('Speedup', tpython / tnumba, 'x')


if __name__ == '__main__':
    main()

