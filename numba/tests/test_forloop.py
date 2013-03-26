#! /usr/bin/env python
# ______________________________________________________________________
'''test_forloop

Test the Numba compiler on a simple for loop over an iterable object.
'''
# ______________________________________________________________________

from numba import *
from numba.testing import test_support

import numpy

import unittest
try:
    import __builtin__ as builtins
except ImportError:
    import builtins

# ______________________________________________________________________

def for_loop_fn_0 (iterable):
    acc = 0.
    for value in iterable:
        acc += value
    return acc

# ______________________________________________________________________

def for_loop_fn_1 (start, stop, inc):
    acc = 0
    for value in range(start, stop, inc):
        acc += value
    return acc

# ______________________________________________________________________

def for_loop_fn_2 (stop):
    acc = 0
    for value_0 in range(stop):
        for value_1 in range(stop):
            acc += value_0 * value_1
    return acc

# ______________________________________________________________________

def for_loop_fn_3 (stop):
    acc = 0
    for i in range(stop):
        for j in range(stop):
            for k in range(stop):
                for l in range(stop):
                    acc += 1
    return acc

# ______________________________________________________________________

def for_loop_w_guard_0 (test_input):
    '''Test case based on issue #25.  See:
    https://github.com/numba/numba/issues/25'''
    acc = 0.0
    for i in range(5):
        if i == test_input:
            acc += 100.0
    return acc

# ______________________________________________________________________

def for_loop_w_guard_1 (test_input):
    '''Test case based on issue #25.  See:
    https://github.com/numba/numba/issues/25'''
    acc = 0.0
    for i in range(5):
        if i == test_input:
            acc += 100.0
        else:
            acc += i
    return acc

# ______________________________________________________________________

def for_loop_fn_4(i, u, p, U):
    '''Test case for issue #48.  See:
    https://github.com/numba/numba/issues/48'''
    s = 0
    t = 0
    for j in range(-p, p+2):
        if U[i+j] == u:
            t = t + 1
        if u == U[i+j]:
            s = s + 1
    if t != s:
        s = -1
    return s

# ______________________________________________________________________

class TestForLoop(unittest.TestCase):
    @test_support.skip_unless(hasattr(builtins, '__noskip__'),
                              "Requires implementation of iteration "
                              "over arrays.")
    def test_compiled_for_loop_fn_0(self):
        test_data = numpy.array([1, 2, 3], dtype = 'l')
        compiled_for_loop_fn = jit(restype=f4,
            argtypes = [i8[:]],backend='ast')(for_loop_fn_0)
        result = compiled_for_loop_fn(test_data)
        self.assertEqual(result, 6)
        self.assertEqual(result, for_loop_fn_0(test_data))

    def test_compiled_for_loop_fn_1(self):
        compiled_for_loop_fn = jit(argtypes = [i4, i4, i4],
                                             restype = i4, backend='ast')(for_loop_fn_1)
        result = compiled_for_loop_fn(1, 4, 1)
        self.assertEqual(result, 6)
        self.assertEqual(result, for_loop_fn_1(1, 4, 1))

    def test_compiled_for_loop_fn_2(self):
        compiled_for_loop_fn = jit(argtypes = [i4],
                                             restype = i4, backend='ast')(for_loop_fn_2)
        result = compiled_for_loop_fn(4)
        self.assertEqual(result, 36)
        self.assertEqual(result, for_loop_fn_2(4))

    def test_compiled_for_loop_fn_3(self):
        compiled_for_loop_fn = jit(argtypes = [i4],
                                             restype = i4, backend='ast')(for_loop_fn_3)
        result = compiled_for_loop_fn(3)
        self.assertEqual(result, for_loop_fn_3(3))
        self.assertEqual(result, 81)

    def test_compiled_for_loop_w_guard_0(self):
        compiled_for_loop_w_guard = autojit(backend='ast')(for_loop_w_guard_0)
        self.assertEqual(compiled_for_loop_w_guard(5.),
                         for_loop_w_guard_0(5.))
        self.assertEqual(compiled_for_loop_w_guard(4.),
                         for_loop_w_guard_0(4.))

    def test_compiled_for_loop_w_guard_1(self):
        compiled_for_loop_w_guard = autojit(backend='ast')(for_loop_w_guard_1)
        self.assertEqual(compiled_for_loop_w_guard(5.),
                         for_loop_w_guard_1(5.))
        self.assertEqual(compiled_for_loop_w_guard(4.),
                         for_loop_w_guard_1(4.))

    def test_compiled_for_loop_fn_4(self):
        compiled = jit('i4(i4,f8,i4,f8[:])')(for_loop_fn_4)
        args0 = 5, 1.0, 2, numpy.ones(10)
        self.assertEqual(compiled(*args0), for_loop_fn_4(*args0))
        args1 = 5, 1.0, 2, numpy.zeros(10)
        self.assertEqual(compiled(*args1), for_loop_fn_4(*args1))

# ______________________________________________________________________

if __name__ == "__main__":
#    compiled_for_loop_fn = jit(argtypes = [i4, i4, i4],
#                               restype = i4, backend='ast', nopython=True)(for_loop_fn_1)
#    result = compiled_for_loop_fn(1, 4, 1)
#    compiled_for_loop_fn = jit(argtypes = [i4],
#                               restype = i4, backend='ast')(for_loop_fn_3)
#    result = compiled_for_loop_fn(3)
#    compiled = jit('i4(i4,f8,i4,f8[:])')(for_loop_fn_4)
#    args0 = 5, 1.0, 2, numpy.ones(10)
#    print compiled(*args0)
#    print for_loop_fn_4(*args0)
    unittest.main()

# ______________________________________________________________________
# End of test_forloop.py
