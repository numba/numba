'''
Example vectorize usage.
'''

import numpy as np
from numba import *
from numbapro.vectorize.basic import BasicVectorize
from numbapro.vectorize.cuda import CudaVectorize
from time import time
from math import sqrt
def discriminant(a, b, c):
    '''a ufunc kernel to compute the discriminant of quadratic equation
    '''
    return sqrt(b ** 2 - 4 * a * c)

def ufunc_discriminant(vecttype=BasicVectorize):
    '''generate a ufunc from discriminant(a, b, c)

    vecttype -- (defaults to BasicVectorize)
                The type of vectorize builder to use.
                For CudaVectorize, the output ufunc can only support float32.
    '''
    vect = vecttype(discriminant)

    if vecttype is CudaVectorize:
        # only works well for float
        supported_types = [f]
    else:
        supported_types = [int32, int64, f, d]
    for ty in supported_types:
        vect.add(restype=ty, argtypes=[ty] * 3)
    return vect.build_ufunc()

