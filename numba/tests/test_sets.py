from __future__ import print_function

import numba.unittest_support as unittest

from collections import namedtuple
import contextlib
import itertools
import math
import sys

import numpy as np

from numba.compiler import compile_isolated, Flags
from numba import jit, types
import numba.unittest_support as unittest
from .support import TestCase, enable_pyobj_flags, nrt_flags, MemoryLeakMixin, tag


def _build_set_literal_usecase(code, args):
    ns = {}
    src = code % {'initializer': ', '.join(repr(arg) for arg in args)}
    code = compile(src, '<>', 'exec')
    eval(code, ns)
    return ns['build_set']

def set_literal_return_usecase(args):
    code = """if 1:
    def build_set():
        return {%(initializer)s}
    """
    return _build_set_literal_usecase(code, args)

def set_literal_convert_usecase(args):
    code = """if 1:
    def build_set():
        my_set = {%(initializer)s}
        return list(my_set)
    """
    return _build_set_literal_usecase(code, args)


def constructor_usecase(arg):
    s = set(arg)
    return len(s)

def iterator_usecase(arg):
    s = set(arg)
    l = []
    for v in s:
        l.append(v)
    return l


needs_set_literals = unittest.skipIf(sys.version_info < (2, 7),
                                     "set literals unavailable before Python 2.7")


class BaseTest(MemoryLeakMixin, TestCase):

    def setUp(self):
        super(BaseTest, self).setUp()
        self.rnd = np.random.RandomState(42)

    def duplicates_array(self, n):
        """
        Get a 1d array with many duplicate values.
        """
        a = np.arange(int(np.sqrt(n)))
        return self.rnd.choice(a, (n,))

    def sparse_array(self, n):
        """
        Get a 1d array with values spread around.
        """
        a = np.arange(n ** 2)
        return self.rnd.choice(a, (n,))


class TestSetLiterals(BaseTest):

    @needs_set_literals
    def test_build_set(self, flags=enable_pyobj_flags):
        pyfunc = set_literal_return_usecase((1, 2, 3, 2))
        self.run_nullary_func(pyfunc, flags=flags)

    @needs_set_literals
    def test_build_heterogenous_set(self, flags=enable_pyobj_flags):
        pyfunc = set_literal_return_usecase((1, 2.0, 3j, 2))
        self.run_nullary_func(pyfunc, flags=flags)
        # Check that items are inserted in the right order (here the
        # result will be {2}, not {2.0})
        pyfunc = set_literal_return_usecase((2.0, 2))
        got, expected = self.run_nullary_func(pyfunc, flags=flags)
        self.assertIs(type(got.pop()), type(expected.pop()))

    @needs_set_literals
    def test_build_set_nopython(self):
        arg = list(self.sparse_array(50))
        #arg = (1, 2, 3, 42, 5, 3)
        pyfunc = set_literal_convert_usecase(arg)
        cfunc = jit(nopython=True)(pyfunc)

        expected = pyfunc()
        got = cfunc()
        self.assertPreciseEqual(sorted(expected), sorted(got))


class TestSets(BaseTest):

    def test_constructor(self):
        pyfunc = constructor_usecase
        cfunc = jit(nopython=True)(pyfunc)
        def check(arg):
            self.assertPreciseEqual(pyfunc(arg), cfunc(arg))

        check((1, 2, 3, 2, 7))
        check(self.duplicates_array(200))
        check(self.sparse_array(200))

    def test_iterator(self):
        pyfunc = iterator_usecase
        cfunc = jit(nopython=True)(pyfunc)
        def check(arg):
            self.assertPreciseEqual(sorted(pyfunc(arg)),
                                    sorted(cfunc(arg)))

        check((1, 2, 3, 2, 7))
        check(self.duplicates_array(200))
        check(self.sparse_array(200))


if __name__ == '__main__':
    unittest.main()
