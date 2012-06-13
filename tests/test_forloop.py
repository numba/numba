#! /usr/bin/env python
# ______________________________________________________________________
'''test_forloop

Test the Numba compiler on a simple for loop over an iterable object.
'''
# ______________________________________________________________________

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

class TestForLoop(unittest.TestCase):

    def test_compiled_for_loop_fn_0(self):
        test_data = numpy.array([1, 2, 3], dtype = 'l')
        compiled_for_loop_fn = numba_compile(
            arg_types = [['l']])(for_loop_fn_0)
        result = compiled_for_loop_fn(test_data)
        self.assertEqual(result, 6)
        self.assertEqual(result, for_loop_fn_0(testdata))

    def test_compiled_for_loop_fn_1(self):
        compiled_for_loop_fn = numba_compile(arg_types = ['i','i','i'],
                                             ret_type = 'i')(for_loop_fn_1)
        result = compiled_for_loop_fn(1, 4, 1)
        self.assertEqual(result, 6)
        self.assertEqual(result, for_loop_fn_1(1, 4, 1))

# ______________________________________________________________________

if __name__ == "__main__":
    unittest.main()

# ______________________________________________________________________
# End of test_forloop.py
