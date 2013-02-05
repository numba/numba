# Based upon the version modified by Mark Harris

import numpy as np
import time
import numbapro  # Works without numbapro
from numba import *
from numba import typeof
import unittest

flopsPerInteraction = 30
SOFTENING_SQUARED = 0.01

@jit(void(f4[:], f4[:], f4[:]))
def fast_body_body_interaction(force, pos_mass0, pos_mass1):
    print 'A', typeof(force), typeof(pos_mass0), typeof(pos_mass1)
    r = pos_mass1[:3] - pos_mass0[:3]
    print 'B', typeof(r)
    dist_sqr = (r * r).sum()  # the .sum() returns an object; can we do better.
    print 'C', typeof(dist_sqr)
    s = pos_mass1[3] / dist_sqr
    print 'G', typeof(s)
    # The following raises:
    #   AttributeError: type object 'ObjectType' has no attribute 'itemsize'
    #   The problem seems to be `s` being an object
    force += r * s


if __name__ == "__main__":
    force = np.arange(3, dtype=np.float32)
    posmass0 = np.arange(4, dtype=np.float32)
    posmass1 = np.arange(4, dtype=np.float32)
    fast_body_body_interaction(force, posmass0, posmass1)
