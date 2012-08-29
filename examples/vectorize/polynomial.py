'''
Example vectorize usage.
'''

import numpy as np
from numba import *
from numbapro.vectorize.basic import BasicVectorize
from time import time

def discriminant(a, b, c):
    '''a ufunc kernel to compute the discriminant of quadratic equation
    '''
    return (b ** 2 - 4 * a * c) ** 0.5

def ufunc_discriminant(vecttype=BasicVectorize):
    '''generate a ufunc from discriminant(a, b, c)

    vecttype -- (defaults to BasicVectorize)
                the type of vectorize builder to use.
    '''
    vect = vecttype(discriminant)
    supported_types = (int32, int64, f, d)
    for ty in supported_types:
        vect.add(ret_type=ty, arg_types=[ty] * 3)
    return vect.build_ufunc()

