from numbapro.vectorize import basic, parallel, stream
from numba import *

def add(a, b):
    return a + b

bv = basic.BasicVectorize(add)
bv.add(ret_type=d, arg_types=[d, d])
b_add_d = bv.build_ufunc()

sv = stream.StreamVectorize(add)
sv.add(ret_type=d, arg_types=[d, d])
s_add_d = sv.build_ufunc()

pv = parallel.ParallelVectorize(add)
pv.add(ret_type=d, arg_types=[d, d])
p_add_d = pv.build_ufunc()

