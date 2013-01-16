import time

import numbapro
import numba
from numba import *
import numpy as np

# Bug reported by Jeremiah L. Lowin

def timer(pyfunc, numbafunc, *args, **kwargs):
    t1 = time.time()
    pyresult = pyfunc(*args, **kwargs)
    t2 = time.time()
    print ''
    print 'python function took: {0}'.format(t2-t1)

    t3 = time.time()
    numbaresult = numbafunc(*args, **kwargs)
    t4 = time.time()
    print 'numba function took: {0}'.format(t4-t3)
    print 'speedup: {0}x'.format(np.round((t2-t1) / (t4-t3),2))

    if not np.allclose(pyresult, numbaresult):
        print '======='
        print 'WARNING: python and numba results are different!!'
        print '======='

    print ''

def vmdot(x, w):
    out = np.empty((x.shape[0], w.shape[1]))
    for i in range(x.shape[0]):
        dot_prod = np.dot(x[i], w)
        out[i] = np.exp(-1 * dot_prod)

    return out

def test_vmdot():
    numba_vmdot = jit(restype=double[:,:],
                      argtypes=[double[:,:], double[:,:]])(vmdot)
    x = np.random.random((1000, 1000))
    w = np.random.random((1000, 1000)) / 1000.
    timer(vmdot, numba_vmdot, x, w) # fails without error -- different results

if __name__ == '__main__':
    test_vmdot()
