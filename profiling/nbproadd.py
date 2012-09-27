from numbapro.vectorize import basic, parallel, stream
from numba import *

def add(a, b):
    return a + b

bv = basic.BasicVectorize(add)
bv.add(restype=d, argtypes=[d, d])
b_add_d = bv.build_ufunc()

sv = stream.StreamVectorize(add)
sv.add(restype=d, argtypes=[d, d])
s_add_d = sv.build_ufunc()

pv = parallel.ParallelVectorize(add)
pv.add(restype=d, argtypes=[d, d])
p_add_d = pv.build_ufunc()

