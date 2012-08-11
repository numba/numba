#! /usr/bin/env python
# ______________________________________________________________________

from numba.decorators import jit

import numpy

import unittest

# ______________________________________________________________________

def while_loop_fn_0(max_index, indexable):
    i = 0
    acc = 0.
    while i < max_index:
        acc += indexable[i]
        i += 1
    return acc

# ______________________________________________________________________

def while_loop_fn_1(indexable):
    i = 0
    acc = 0.
    while i < len(indexable):
        acc += indexable[i]
        i += 1
    return acc

# ______________________________________________________________________

def while_loop_fn_2(ndarr):
    i = 0
    acc = 0.
    while i < ndarr.shape[0]:
        acc += ndarr[i]
        i += 1
    return acc

# ______________________________________________________________________

def while_loop_fn_3(count):
    i = 0
    acc = 1.
    while i < count:
        acc *= 2
        i += 1
    return acc

# ______________________________________________________________________

def while_loop_fn_4(start, stop, inc):
    '''Intended to parallel desired translation target for
    test_forloop.for_loop_fn_1.'''
    acc = 0
    i = start
    while i != stop:
        acc += i
        i += inc
    return acc

# ______________________________________________________________________

def while_loop_fn_5(i_max, j_max):
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

class TestWhile(unittest.TestCase):
    def _do_test(self, function, arg_types, *args, **kws):
        _jit = (jit(arg_types = arg_types)
                   if arg_types is not None else jit())
        compiled_fn = _jit(function)
        self.assertEqual(compiled_fn(*args, **kws), function(*args, **kws))

    def test_while_loop_fn_0(self):
        test_data = numpy.array([1., 2., 3.])
        self._do_test(while_loop_fn_0, ['l', ['d']], len(test_data), test_data)

    def test_while_loop_fn_1(self):
        self._do_test(while_loop_fn_1, [['d']], numpy.array([1., 2., 3.]))

    def test_while_loop_fn_2(self):
        self._do_test(while_loop_fn_2, [['d']], numpy.array([1., 2., 3.]))

    def test_while_loop_fn_3(self):
        compiled_fn = jit(arg_types = ['l'])(while_loop_fn_3)
        compiled_result = compiled_fn(3)
        self.assertEqual(compiled_result, while_loop_fn_3(3))
        self.assertEqual(compiled_result, 8.)

    def test_while_loop_fn_4(self):
        compiled_fn = jit(arg_types = ['l', 'l', 'l'],
                                    ret_type = 'l')(while_loop_fn_4)
        compiled_result = compiled_fn(1, 4, 1)
        self.assertEqual(compiled_result, while_loop_fn_4(1, 4, 1))
        self.assertEqual(compiled_result, 6)

    def test_while_loop_fn_5(self):
        compiled_fn = jit(arg_types = ['d', 'd'])(while_loop_fn_5)
        compiled_result = compiled_fn(3, 4)
        self.assertEqual(compiled_result, while_loop_fn_5(3, 4))
        self.assertEqual(compiled_result, 18.)

# ______________________________________________________________________

if __name__ == "__main__":
    unittest.main()

# ______________________________________________________________________
# End of test_while.py
