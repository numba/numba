import numpy as np
from numba import *
from numbapro.vectorize import Vectorize

def vector_add(a, b):
    return a + b

def main(backend):
    # build basic native code ufunc
    bv = Vectorize(vector_add, target='cpu', backend=backend)
    bv.add(restype=int32,  argtypes=[int32, int32])
    bv.add(restype=uint32, argtypes=[uint32, uint32])
    bv.add(restype=f4,     argtypes=[f4, f4])
    bv.add(restype=f8,     argtypes=[f8, f8])
    functions = bv.build_ufunc_core()
    print functions[0][0]

if __name__ == '__main__':
    main('ast')

