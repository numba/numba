# -*- coding: utf-8 -*-
"""
Example of multithreading by releasing the GIL through ctypes.
"""
from __future__ import print_function, division, absolute_import

from timeit import repeat
import threading
from ctypes import pythonapi, c_void_p
from math import exp

import numpy as np
from numba import jit, void, double

nthreads = 2
size = 1e6

def timefunc(correct, s, func, *args, **kwargs):
    print(s.ljust(20), end=" ")
    # Make sure the function is compiled before we start the benchmark
    res = func(*args, **kwargs)
    if correct is not None:
        assert np.allclose(res, correct)
    # time it
    print('{:>5.0f} ms'.format(min(repeat(lambda: func(*args, **kwargs),
                                          number=5, repeat=2)) * 1000))
    return res

def make_singlethread(inner_func):
    def func(*args):
        length = len(args[0])
        result = np.empty(length, dtype=np.float64)
        inner_func(result, *args)
        return result
    return func

def make_multithread(inner_func, numthreads):
    def func_mt(*args):
        length = len(args[0])
        result = np.empty(length, dtype=np.float64)
        args = (result,) + args        
        chunklen = (length + 1) // numthreads
        chunks = [[arg[i * chunklen:(i + 1) * chunklen] for arg in args]
                  for i in range(numthreads)]

        # You should make sure inner_func is compiled at this point, because
        # the compilation must happen on the main thread. This is the case
        # in this example because we use jit().
        threads = [threading.Thread(target=inner_func, args=chunk)
                   for chunk in chunks[:-1]]
        for thread in threads:
            thread.start()

        # the main thread handles the last chunk
        inner_func(*chunks[-1])

        for thread in threads:
            thread.join()
        return result
    return func_mt
  
savethread = pythonapi.PyEval_SaveThread
savethread.argtypes = []
savethread.restype = c_void_p

restorethread = pythonapi.PyEval_RestoreThread
restorethread.argtypes = [c_void_p]
restorethread.restype = None

def inner_func(result, a, b):
    threadstate = savethread()
    for i in range(len(result)):
        result[i] = exp(2.1 * a[i] + 3.2 * b[i])
    restorethread(threadstate)

signature = void(double[:], double[:], double[:])
inner_func_nb = jit(signature, nopython=True)(inner_func)
func_nb = make_singlethread(inner_func_nb)
func_nb_mt = make_multithread(inner_func_nb, nthreads)
            
def func_np(a, b):
    return np.exp(2.1 * a + 3.2 * b)

a = np.random.rand(size)
b = np.random.rand(size)
c = np.random.rand(size)

correct = timefunc(None, "numpy (1 thread)", func_np, a, b)
timefunc(correct, "numba (1 thread)", func_nb, a, b)
timefunc(correct, "numba (%d threads)" % nthreads, func_nb_mt, a, b)
