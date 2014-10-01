from __future__ import print_function, absolute_import
import numpy
import numba.unittest_support as unittest
from numba.compiler import compile_isolated
from numba import types, typeof


def return_double_or_none(x):
    if x:
        ret = None
    else:
        ret = 1.2
    return ret


def return_different_statement(x):
    if x:
        return None
    else:
        return 1.2


def is_this_a_none(x):
    if x:
        val_or_none = None
    else:
        val_or_none = x

    if val_or_none is None:
        return x - 1

    if val_or_none is not None:
        return x + 1


def a_is_b(a, b):
    """
    Note in nopython mode, this operation does not make much sense.
    Because we don't have objects anymore.
    `a is b` is always False if not operating on None and Optional type
    """
    return a is b


def a_is_not_b(a, b):
    """
    This is `not (a is b)`
    """
    return a is not b


class TestOptional(unittest.TestCase):
    def test_return_double_or_none(self):
        pyfunc = return_double_or_none
        cres = compile_isolated(pyfunc, [types.boolean])
        cfunc = cres.entry_point

        for v in [True, False]:
            self.assertEqual(pyfunc(v), cfunc(v))

    def test_return_different_statement(self):
        pyfunc = return_different_statement
        cres = compile_isolated(pyfunc, [types.boolean])
        cfunc = cres.entry_point

        for v in [True, False]:
            self.assertEqual(pyfunc(v), cfunc(v))

    def test_is_this_a_none(self):
        pyfunc = is_this_a_none
        cres = compile_isolated(pyfunc, [types.intp])
        cfunc = cres.entry_point

        for v in [-1, 0, 1, 2]:
            self.assertEqual(pyfunc(v), cfunc(v))

    def test_a_is_b_intp(self):
        pyfunc = a_is_b
        cres = compile_isolated(pyfunc, [types.intp, types.intp])
        cfunc = cres.entry_point
        self.assertFalse(cfunc(1, 1))

    def test_a_is_b_array(self):
        pyfunc = a_is_b
        ary = numpy.arange(2)
        aryty = typeof(ary)
        cres = compile_isolated(pyfunc, [aryty, aryty])
        cfunc = cres.entry_point
        self.assertFalse(cfunc(ary, ary))

    def test_a_is_not_b_intp(self):
        pyfunc = a_is_not_b
        cres = compile_isolated(pyfunc, [types.intp, types.intp])
        cfunc = cres.entry_point
        self.assertTrue(cfunc(1, 1))

    def test_a_is_not_b_array(self):
        pyfunc = a_is_not_b
        ary = numpy.arange(2)
        aryty = typeof(ary)
        cres = compile_isolated(pyfunc, [aryty, aryty])
        cfunc = cres.entry_point
        self.assertTrue(cfunc(ary, ary))


if __name__ == '__main__':
    unittest.main()
