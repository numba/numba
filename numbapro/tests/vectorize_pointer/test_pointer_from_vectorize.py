import numpy as np
from numba import *
from numbapro.vectorize import Vectorize
from ctypes import *
from timeit import default_timer as time

def vector_add(a, b):
    return a + b

def test(backend, target):
    print 'testing', backend, target
    ts = time()
    # build basic native code ufunc
    bv = Vectorize(vector_add, target='stream', backend=backend)
    bv.add(restype=int32,  argtypes=[int32, int32])
    #bv.add(restype=uint32, argtypes=[uint32, uint32])
    #bv.add(restype=f4,     argtypes=[f4, f4])
    #bv.add(restype=f8,     argtypes=[f8, f8])
    functions = bv.build_ufunc_core()

    lfunc, ptr = functions[0]

    # Make ctypes callable
    # void (*)(void** args, py_ssize_t* dimensions, py_ssize_t* steps, void* data)

    void_p2p = POINTER(c_void_p)
    py_ssize_t = np.ctypeslib.c_intp
    py_ssize_t_p = POINTER(py_ssize_t)

    prototype = CFUNCTYPE(None, void_p2p, py_ssize_t_p, py_ssize_t_p, c_void_p)
    callable = prototype(ptr)

    # test calling it

    A = np.arange(1024)
    B = A
    C = A.copy()

    args = (c_void_p * 3)(A.ctypes.data, # input
                          B.ctypes.data, # input
                          C.ctypes.data) # output always goes last
    dimensions = (py_ssize_t * 1)(A.shape[0])
    steps = (py_ssize_t * 3)(A.strides[0], B.strides[0], C.strides[0])
    data = None

    callable(args, dimensions, steps, data)

    te = time()

    print C
    print 'Completed in %f s' % (te - ts)

    assert (C == A + B).all()
    print 'ok'
    print

def main():
    backend = 'ast'
    for target in ['cpu', 'stream', 'parallel']:
        test(backend, target)

if __name__ == '__main__':
    main()
