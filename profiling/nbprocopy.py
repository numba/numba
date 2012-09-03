from numbapro.vectorize import basic, parallel, stream
from numba import *

def copy(src):
    return src

bv = basic.BasicVectorize(copy)
bv.add(ret_type=d, arg_types=[d])
b_copy_d = bv.build_ufunc()

sv = stream.StreamVectorize(copy)
sv.add(ret_type=d, arg_types=[d])
s_copy_d = sv.build_ufunc()

pv = parallel.ParallelVectorize(copy)
pv.add(ret_type=d, arg_types=[d])
p_copy_d = pv.build_ufunc()

