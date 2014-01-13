import time

import numba
from numba import *
import numpy as np

# Bug reported by Jeremiah L. Lowin

def timer(pyfunc, numbafunc, *args, **kwargs):
    t1 = time.time()
    pyresult = pyfunc(*args, **kwargs)
    t2 = time.time()
    print(('python function took: {0}'.format(t2-t1)))

    t3 = time.time()
    numbaresult = numbafunc(*args, **kwargs)
    t4 = time.time()
    print(('numba function took: {0}'.format(t4-t3)))
    print(('speedup: {0}x'.format(np.round((t2-t1) / (t4-t3),2))))

    assert np.allclose(pyresult, numbaresult)


def timer2(pyfunc, numbafunc, *args, **kwargs):
    t1 = time.time()
    pyresult = np.empty_like(args[0])
    pyargs = args + (pyresult,)
    pyfunc(*pyargs, **kwargs)
    t2 = time.time()
    print(('python function took: {0}'.format(t2-t1)))

    t3 = time.time()
    numbaresult = np.empty_like(args[0])
    nbargs = args + (numbaresult,)
    numbafunc(*nbargs, **kwargs)
    t4 = time.time()
    print(('numba function took: {0}'.format(t4-t3)))
    print(('speedup: {0}x'.format(np.round((t2-t1) / (t4-t3),2))))

    assert np.allclose(pyresult, numbaresult)


def vmdot(x, w):
    out = np.empty((x.shape[0], w.shape[1]))
    for i in range(x.shape[0]):
        dot_prod = np.dot(x[i], w)
        out[i] = np.exp(-1 * dot_prod)

    return out

def vmdot2(x, w, out):
    for i in range(x.shape[0]):
        dot_prod = np.dot(x[i], w)
        out[i] = np.exp(-1 * dot_prod)


def test_vmdot():
    numba_vmdot = jit(restype=double[:,:],
                      argtypes=[double[:,:], double[:,:]])(vmdot)
    x = np.random.random((1000, 1000))
    w = np.random.random((1000, 1000)) / 1000.
    timer(vmdot, numba_vmdot, x, w)

    # ensure this compiles
    numba_vmdot2 = jit(argtypes=[double[:,:], double[:,:], double[:,:]])(vmdot2)

    timer2(vmdot2, numba_vmdot2, x, w)

if __name__ == '__main__':
    test_vmdot()
