from numba import autojit
import numpy as np
import unittest

@autojit
def for_loop_fn_1 (start, stop, inc):
    acc = 0
    for value in range(start, stop, inc):
        acc += value
    return acc

@autojit
def for_loop_fn_1a (start, stop):
    acc = 0
    for value in range(start, stop):
        acc += value
    return acc

@autojit
def for_loop_fn_1b (stop):
    acc = 0
    for value in range(stop):
        acc += value
    return acc


class TestForLoop(unittest.TestCase):
    def test_compiled_for_loop_fn_many(self):
        for lo in xrange( -10, 11 ):
            for hi in xrange( -10, 11 ):
                for step in xrange( -20, 21 ):
                    if step:
                        self.assertEqual(for_loop_fn_1(lo, hi, step),
                                         for_loop_fn_1.py_func(lo, hi, step),
                                         'failed for %d/%d/%d' % (lo, hi, step))
                        self.assertEqual(for_loop_fn_1a(lo, hi),
                                         for_loop_fn_1a.py_func(lo, hi),
                                         'failed for %d/%d' % (lo, hi))
                        self.assertEqual(for_loop_fn_1b(hi),
                                         for_loop_fn_1b.py_func(hi),
                                         'failed for %d' % hi)