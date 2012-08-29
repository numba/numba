import numpy as np
from numba import *
from numbapro.vectorize.basic import BasicVectorize

def vector_power(a, b):
    return a ** b

pv = BasicVectorize(vector_power)
pv.add(ret_type=int32, arg_types=[int32, int32])
para_ufunc = pv.build_ufunc()

a = np.arange(10, dtype=np.int32)
b = np.arange(10, dtype=np.int32)

result = para_ufunc(a, b)
print result
