# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import

import numba
import numpy

n = 50
steps = 10000

@numba.autojit()
def calc():
    value = numpy.zeros((n,n))

    for i in range(n):
        for j in range(n):
            value[i][j] += 1.0

    return value

def leaky():
    for i in range(steps):
        value = calc()

leaky()
