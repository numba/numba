# Used in test_jit_module for testing jit_module with implicit device=True

from numba import cuda
import numpy as np


# Bare functions that should be jitted as device functions

def inc(x):
    return x + 1


def add(x, y):
    return x + y


def inc_add(x):
    y = inc(x)
    return add(x, y)


# Objects that should not get jitted

mean = np.mean


class Foo(object):
    pass


# Kernels needed for testing the device functions jitted by jit_module

@cuda.jit
def call_device_inc(x):
    x[0] = inc(x[0])


@cuda.jit
def call_device_add(x, y):
    x[0] = add(x[0], y[0])


@cuda.jit
def call_device_inc_add(x):
    x[0] = inc_add(x[0])


# Host functions to call kernels, hidden in a class to protect them from
# jit_module

class Callers:
    @classmethod
    def inc(cls, x):
        x = np.array([x])
        call_device_inc[1, 1](x)
        return x[0]

    @classmethod
    def add(cls, x, y):
        x = np.array([x])
        y = np.array([y])
        call_device_add[1, 1](x, y)
        return x[0]

    @classmethod
    def inc_add(cls, x):
        x = np.array([x])
        call_device_inc_add[1, 1](x)
        return x[0]


cuda.jit_module()
