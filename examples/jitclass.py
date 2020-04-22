#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A simple jitclass example.
"""

import typing

import numpy as np
from numba import float32
from numba.experimental import jitclass


@jitclass([("array", float32[:])])
class Bag(object):
    # For simple scalar fields, we can infer the spec from type annotations.
    value: int
    # For numpy arrays, we need to specify type and rank in the spec explicitly.
    array: np.ndarray

    def __init__(self, value: int):
        self.value = value
        self.array = np.zeros(value, dtype=np.float32)

    @property
    def size(self) -> int:
        return self.array.size

    def increment(self, val: typing.Union[int, float]) -> np.ndarray:
        for i in range(self.size):
            self.array[i] += val
        return self.array


mybag = Bag(7)
print('isinstance(mybag, Bag)', isinstance(mybag, Bag))
print('mybag.value', mybag.value)
print('mybag.array', mybag.array)
print('mybag.size', mybag.size)
print('mybag.increment(2)', mybag.increment(2)) # call with int
print('mybag.increment(3.14)', mybag.increment(3.14)) # call with float
