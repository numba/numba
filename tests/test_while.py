#! /usr/bin/env python
# ______________________________________________________________________

from numba.decorators import numba_compile, function

import numpy
import numpy as np

import unittest

# ______________________________________________________________________

def _while_loop_fn_0(max_index, indexable):
    i = 0
    acc = 0.
    while i < max_index:
        acc += indexable[i]
        i += 1
    return acc

# ______________________________________________________________________

def _while_loop_fn_1(indexable):
    i = 0
    acc = 0.
    while i < len(indexable):
        acc += indexable[i]
        i += 1
    return acc

# ______________________________________________________________________

def _while_loop_fn_2(ndarr):
    i = 0
    acc = 0.
    while i < ndarr.shape[0]:
        acc += ndarr[i]
        i += 1
    return acc

# ______________________________________________________________________

def _while_loop_fn_3(count):
    i = 0
    acc = 1.
    while i < count:
        acc *= 2
        i += 1
    return acc

# ______________________________________________________________________

def _while_loop_fn_4(start, stop, inc):
    '''Intended to parallel desired translation target for
    test_forloop.for_loop_fn_1.'''
    acc = 0
    i = start
    while i != stop:
        acc += i
        i += inc
    return acc

# ______________________________________________________________________

def _while_loop_fn_5(i_max, j_max):
    j = 1.
    acc = 0.
    while j < j_max:
        i = 1.
        while i < i_max:
            acc += i * j
            i += 1.
        j += 1.
    return acc

# ______________________________________________________________________

while_loop_fn_0 = function(_while_loop_fn_0)
while_loop_fn_1 = function(_while_loop_fn_1)
while_loop_fn_2 = function(_while_loop_fn_2)
while_loop_fn_3 = function(_while_loop_fn_3)
while_loop_fn_4 = function(_while_loop_fn_4)
while_loop_fn_5 = function(_while_loop_fn_5)

class TestWhile(unittest.TestCase):
    def _do_test(self, name, *args, **kws):
        compiled = globals()[name]
        uncompiled = globals()['_' + name]
        self.assertEqual(compiled(*args, **kws), uncompiled(*args, **kws))

    def test_while_loop_fn_0(self):
        test_data = numpy.array([1., 2., 3.])
        self._do_test('while_loop_fn_0', len(test_data), test_data)

    def test_while_loop_fn_1(self):
        self._do_test('while_loop_fn_1', numpy.array([1., 2., 3.]))

    def test_while_loop_fn_2(self):
        self._do_test('while_loop_fn_2', numpy.array([1., 2., 3.]))

    def test_while_loop_fn_3(self):
        self._do_test('while_loop_fn_3', 3)

    def test_while_loop_fn_4(self):
        self._do_test('while_loop_fn_4', 1, 4, 1)

    def test_while_loop_fn_5(self):
        self._do_test('while_loop_fn_5', 3, 4)

# ______________________________________________________________________

if __name__ == "__main__":
    unittest.main()

# ______________________________________________________________________
# End of test_while.py
