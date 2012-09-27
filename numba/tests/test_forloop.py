#! /usr/bin/env python
# ______________________________________________________________________
'''test_forloop

Test the Numba compiler on a simple for loop over an iterable object.
'''
# ______________________________________________________________________

from numba.decorators import jit

import numpy

import unittest
import __builtin__

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

class TestForLoop(unittest.TestCase):
    @unittest.skipUnless(hasattr(__builtin__, '__noskip__'), 
                         "Requires implementation of iteration " 
                         "over arrays.")
    def test_compiled_for_loop_fn_0(self):
        test_data = numpy.array([1, 2, 3], dtype = 'l')
        compiled_for_loop_fn = jit(
            argtypes = [['l']])(for_loop_fn_0)
        result = compiled_for_loop_fn(test_data)
        self.assertEqual(result, 6)
        self.assertEqual(result, for_loop_fn_0(testdata))

    def test_compiled_for_loop_fn_1(self):
        compiled_for_loop_fn = jit(argtypes = ['i','i','i'],
                                             restype = 'i')(for_loop_fn_1)
        result = compiled_for_loop_fn(1, 4, 1)
        self.assertEqual(result, 6)
        self.assertEqual(result, for_loop_fn_1(1, 4, 1))

    def test_compiled_for_loop_fn_2(self):
        compiled_for_loop_fn = jit(argtypes = ['i'],
                                             restype = 'i')(for_loop_fn_2)
        result = compiled_for_loop_fn(4)
        self.assertEqual(result, 36)
        self.assertEqual(result, for_loop_fn_2(4))

    def test_compiled_for_loop_fn_3(self):
        compiled_for_loop_fn = jit(argtypes = ['i'],
                                             restype = 'i')(for_loop_fn_3)
        result = compiled_for_loop_fn(3)
        self.assertEqual(result, for_loop_fn_3(3))
        self.assertEqual(result, 81)

    def test_compiled_for_loop_w_guard_0(self):
        compiled_for_loop_w_guard = jit()(for_loop_w_guard_0)
        self.assertEqual(compiled_for_loop_w_guard(5.),
                         for_loop_w_guard_0(5.))
        self.assertEqual(compiled_for_loop_w_guard(4.),
                         for_loop_w_guard_0(4.))

    def test_compiled_for_loop_w_guard_1(self):
        compiled_for_loop_w_guard = jit()(for_loop_w_guard_1)
        self.assertEqual(compiled_for_loop_w_guard(5.),
                         for_loop_w_guard_1(5.))
        self.assertEqual(compiled_for_loop_w_guard(4.),
                         for_loop_w_guard_1(4.))

# ______________________________________________________________________

if __name__ == "__main__":
    unittest.main()

# ______________________________________________________________________
# End of test_forloop.py
