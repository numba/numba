from numbapro.vectorize import basic, parallel, stream
from numba import *
from math import sqrt

def poly(a, b, c):
    return sqrt(b**2 + 4 * a * c)

bv = basic.BasicVectorize(poly)
bv.add(ret_type=d, arg_types=[d, d, d])
b_poly_d = bv.build_ufunc()

sv = stream.StreamVectorize(poly)
sv.add(ret_type=d, arg_types=[d, d, d])
s_poly_d = sv.build_ufunc()

pv = parallel.ParallelVectorize(poly)
pv.add(ret_type=d, arg_types=[d, d, d])
p_poly_d = pv.build_ufunc()

