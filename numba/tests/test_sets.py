from __future__ import print_function

import numba.unittest_support as unittest

from collections import namedtuple
import contextlib
import itertools
import math
import sys

from numba.compiler import compile_isolated, Flags
from numba import jit, types
import numba.unittest_support as unittest
from .support import TestCase, enable_pyobj_flags, MemoryLeakMixin, tag


def build_set_literal_usecase(*args):
    ns = {}
    src = """if 1:
    def build_set():
        return {%s}
    """ % ', '.join(repr(arg) for arg in args)
    code = compile(src, '<>', 'exec')
    eval(code, ns)
    return ns['build_set']


def set_constructor_usecase(arg):
    s = set(arg)
    return len(s)


needs_set_literals = unittest.skipIf(sys.version_info < (2, 7),
                                     "set literals unavailable before Python 2.7")

class TestSetLiterals(TestCase):

    @needs_set_literals
    def test_build_set(self, flags=enable_pyobj_flags):
        pyfunc = build_set_literal_usecase(1, 2, 3, 2)
        self.run_nullary_func(pyfunc, flags=flags)

    @needs_set_literals
    def test_build_heterogenous_set(self, flags=enable_pyobj_flags):
        pyfunc = build_set_literal_usecase(1, 2.0, 3j, 2)
        self.run_nullary_func(pyfunc, flags=flags)
        # Check that items are inserted in the right order (here the
        # result will be {2}, not {2.0})
        pyfunc = build_set_literal_usecase(2.0, 2)
        got, expected = self.run_nullary_func(pyfunc, flags=flags)
        self.assertIs(type(got.pop()), type(expected.pop()))


class TestSets(MemoryLeakMixin, TestCase):

    def test_set_constructor(self):
        pyfunc = set_constructor_usecase
        cfunc = jit(nopython=True)(pyfunc)
        arg = (1, 2, 3, 2, 7)
        self.assertPreciseEqual(pyfunc(arg), cfunc(arg))


if __name__ == '__main__':
    unittest.main()
