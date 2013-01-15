import numbapro
import numba
from numba import *
import numpy as np

# Bug reported by Jeremiah L. Lowin
def vmdot(x, w):
    out = np.empty((x.shape[0], w.shape[1]))
    print numba.typeof(out)
    for i in range(x.shape[0]):
        dot_prod = np.dot(x[i], w)
        print numba.typeof(dot_prod)
        print numba.typeof(np.exp(-1 * dot_prod))
        out[i] = np.exp(-1 * dot_prod)
    return out

def test_vmdot():
    numba_vmdot = jit(restype=double[:,:], argtypes=[double[:,:], double[:,:]])(vmdot)
    x = np.random.random((1000, 1000))
    w = np.random.random((1000, 1000)) / 1000.
    print numba_vmdot(x, w)
    #timer(vmdot, numba_vmdot, x, w) # fails without error -- different results

if __name__ == '__main__':
    test_vmdot()