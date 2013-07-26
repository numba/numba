# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

from   numba import int_, jit
import numpy

@jit(argtypes=(int_[:],))
def test1(arr):
    u = 0
    for x in arr:
        u += len(arr)

    v = 0
    for y in arr:
        v += len(arr)

    return u + v

@jit(argtypes=(int_[:],))
def test2(arr):
    s = 0
    for i, x in enumerate(arr):
        s += i*x

    s2 = 0
    for i2, x2 in enumerate(arr, 1):
        s2 += i2*x2
    return s+s2

if __name__ == '__main__':
    arr = numpy.arange(1, 4)
    assert test1(arr) == test1.py_func(arr)
    assert test2(arr) == test2.py_func(arr)