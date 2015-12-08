#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

from timeit import default_timer as timer

import numpy as np

from numba import jit


def bubblesort(X):
    N = X.shape[0]
    for end in range(N, 1, -1):
        for i in range(end - 1):
            if X[i] > X[i + 1]:
                X[i], X[i + 1] = X[i + 1], X[i]


bubblesort_fast = jit(nopython=True)(bubblesort)

dtype = np.int64

def main():
    Xtest = np.array(list(reversed(range(8))), dtype=dtype)

    print('== Test Pure-Python ==')
    X0 = Xtest.copy()
    bubblesort(X0)
    print(X0)

    print('== Test Numba == ')
    X1 = Xtest.copy()
    bubblesort_fast(X1)
    print(X1)

    assert all(X0 == X1)

    REP = 10
    N = 1000

    Xorig = np.array(list(reversed(range(N))), dtype=dtype)

    t0 = timer()
    for t in range(REP):
        X0 = Xorig.copy()
        bubblesort(X0)
    tpython = (timer() - t0) / REP

    t1 = timer()
    for t in range(REP):
        X1 = Xorig.copy()
        bubblesort_fast(X1)
    tnumba = (timer() - t1) / REP

    assert all(X0 == X1)

    print('Python', tpython)
    print('Numba', tnumba)
    print('Speedup', tpython / tnumba, 'x')


if __name__ == '__main__':
    main()
