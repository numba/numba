# Used in test_jit_module for testing jit_module with device=False

from numba import cuda
import numpy as np


# Device functions that should not get jitted by jit_module, and should be
# callable from inc_add when it is jitted as a kernel by jit_module

@cuda.jit(device=True)
def device_inc(x):
    return x + 1


@cuda.jit(device=True)
def device_add(x, y):
    return x + y


# Bare functions that should be jitted as kernels

def inc(x):
    x[0] += 1


def add(x, y):
    x[0] += y[0]


def inc_add(x):
    y = device_inc(x[0])
    x[0] = device_add(x[0], y)


# Objects that should not get jitted

mean = np.mean


class Foo(object):
    pass


# Host functions to call kernels, hidden in a class to protect them from
# jit_module

class Callers:
    @classmethod
    def inc(cls, x):
        x = np.array([x])
        inc[1, 1](x)
        return x[0]

    @classmethod
    def add(cls, x, y):
        x = np.array([x])
        y = np.array([y])
        add[1, 1](x, y)
        return x[0]

    @classmethod
    def inc_add(cls, x):
        x = np.array([x])
        inc_add[1, 1](x)
        return x[0]

    @classmethod
    def py_inc(cls, x):
        x = np.array([x])
        inc.py_func(x)
        return x[0]

    @classmethod
    def py_add(cls, x, y):
        x = np.array([x])
        y = np.array([y])
        add.py_func(x, y)
        return x[0]

    @classmethod
    def py_inc_add(cls, x):
        x = np.array([x])
        # The initial value for y is x, but we don't want to overwrite x in inc
        y = x.copy()

        # We can't call inc_add directly for testing as a Python function,
        # because it calls inc and add - these are Dispatchers, not Python
        # functions. We manually construct the expected result using
        # inc.py_func and add.py_func instead.
        inc.py_func(y)
        add.py_func(x, y)
        return x[0]


cuda.jit_module(device=False)
