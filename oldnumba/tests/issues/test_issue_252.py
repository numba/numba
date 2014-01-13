from numba import *
import numpy as np

@jit(object_(object_[:, :]))
def func(A):
    for x in xrange(2):
        for y in xrange(2):
            items = A[x,y]
            if items == None:
                continue
            for item in items:
                print(item)

    return A