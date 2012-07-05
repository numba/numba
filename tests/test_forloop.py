#! /usr/bin/env python
# ______________________________________________________________________
'''test_forloop

Test the Numba compiler on a simple for loop over an iterable object.
'''
# ______________________________________________________________________

import numba
from numba import *
from numba.decorators import numba_compile

import numpy

import unittest

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

class TestForLoop(unittest.TestCase):
    @unittest.skipUnless(__debug__, "Requires implementation of iteration "
                         "over arrays.")
    def test_compiled_for_loop_fn_0(self):
        test_data = numpy.array([1, 2, 3], dtype=numpy.int32)
        compiled_for_loop_fn = numba_compile(
            ret_type=numba.float32, arg_types=[numba.int32[:]])(for_loop_fn_0)
        result = compiled_for_loop_fn(test_data)
        self.assertEqual(result, 6)
        self.assertEqual(result, for_loop_fn_0(testdata))

    def test_compiled_for_loop_fn_1(self):
        compiled_for_loop_fn = numba_compile(arg_types=[i4, i4, i4],
                                             ret_type=i4)(for_loop_fn_1)
        result = compiled_for_loop_fn(1, 4, 1)
        self.assertEqual(result, 6)
        self.assertEqual(result, for_loop_fn_1(1, 4, 1))

    def test_compiled_for_loop_fn_2(self):
        compiled_for_loop_fn = numba_compile(arg_types=[i4],
                                             ret_type=i4)(for_loop_fn_2)
        result = compiled_for_loop_fn(4)
        self.assertEqual(result, 36)
        self.assertEqual(result, for_loop_fn_2(4))

    def test_compiled_for_loop_fn_3(self):
        compiled_for_loop_fn = numba_compile(arg_types=[i4],
                                             ret_type=i4)(for_loop_fn_3)
        result = compiled_for_loop_fn(3)
        self.assertEqual(result, for_loop_fn_3(3))
        self.assertEqual(result, 81)

# ______________________________________________________________________

if __name__ == "__main__":
    import dis; dis.dis(for_loop_fn_1)
    unittest.main()
    #TestForLoop('test_compiled_for_loop_fn_2').debug()

# ______________________________________________________________________
# End of test_forloop.py
