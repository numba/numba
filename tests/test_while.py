#! /usr/bin/env python
# ______________________________________________________________________

from numba.decorators import numba_compile

import numpy

import sys
import unittest

# ______________________________________________________________________

def while_loop_fn_0(max_index, indexable):
    i = 0
    acc = 0.
    while i < max_index:
        acc += indexable[i]
    return acc

# ______________________________________________________________________

def while_loop_fn_1(indexable):
    i = 0
    acc = 0.
    while i < len(indexable):
        acc += indexable[i]
    return acc

# ______________________________________________________________________

class TestWhile(unittest.TestCase):
    def _do_test(self, function, arg_types, *args, **kws):
        _numba_compile = (numba_compile(arg_types = arg_types)
                          if arg_types is not None else numba_compile())
        compiled_fn = _numba_compile(function)
        self.assertEqual(compiled_fn(*args, **kws), function(*args, **kws))

    def test_while_loop_fn_0(self):
        test_data = numpy.array([1., 2., 3.])
        self._do_test(while_loop_fn_0, ['l', ['d']], len(test_data), test_data)

    def test_while_loop_fn_1(self):
        self._do_test(while_loop_fn_1, [['d']], numpy.array([1., 2., 3.]))

# ______________________________________________________________________

if __name__ == "__main__":
    unittest.main(*sys.argv[1:])

# ______________________________________________________________________
# End of test_while.py
