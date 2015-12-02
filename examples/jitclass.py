#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A simple jitclass example.
"""

import numpy as np
from numba import jitclass          # import the decorator
from numba import int32, float32    # import the types

spec = [
    ('value', int32),               # a simple scalar field
    ('array', float32[:]),          # an array field
]


@jitclass(spec)
class Bag(object):
    def __init__(self, value):
        self.value = value
        self.array = np.zeros(value, dtype=np.float32)

    @property
    def size(self):
        return self.array.size

    def increment(self, val):
        for i in range(self.size):
            self.array[i] += val
        return self.array


mybag = Bag(21)
print('isinstance(mybag, Bag)', isinstance(mybag, Bag))
print('mybag.value', mybag.value)
print('mybag.array', mybag.array)
print('mybag.size', mybag.size)
print('mybag.increment(3)', mybag.increment(3))
print('mybag.increment(6)', mybag.increment(6))
