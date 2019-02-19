from __future__ import print_function

import math
import re
import textwrap
import operator

import numpy as np

import numba.unittest_support as unittest
from numba.compiler import compile_isolated
from numba import jit, types
from numba.errors import TypingError
from .support import TestCase


def what():
    pass

def foo():
    return what()

def bar(x):
    return x.a

def issue_868(a):
    return a.shape * 2

def impossible_return_type(x):
    if x > 0:
        return ()
    else:
        return 1j

def bad_hypot_usage():
    return math.hypot(1)

def imprecise_list():
    l = []
    return len(l)

def using_imprecise_list():
    a = np.array([])
    return a.astype(np.int32)

def unknown_module():
    return numpyz.int32(0)

def nop(x, y, z):
    pass

def array_setitem_invalid_cast():
    arr = np.empty(1, dtype=np.float64)
    arr[0] = 1j  # invalid cast from complex to float
    return arr


class Foo(object):
    def __repr__(self):
        return "<Foo instance>"


class TestTypingError(unittest.TestCase):

    def test_unknown_function(self):
        try:
            compile_isolated(foo, ())
        except TypingError as e:
            self.assertIn("Untyped global name 'what'", str(e))
        else:
            self.fail("Should raise error")

    def test_unknown_attrs(self):
        try:
            compile_isolated(bar, (types.int32,))
        except TypingError as e:
            self.assertIn("Unknown attribute 'a' of type int32", str(e))
        else:
            self.fail("Should raise error")

    def test_unknown_module(self):
        # This used to print "'object' object has no attribute 'int32'"
        with self.assertRaises(TypingError) as raises:
            compile_isolated(unknown_module, ())
        self.assertIn("Untyped global name 'numpyz'", str(raises.exception))

    def test_issue_868(self):
        '''
        Summary: multiplying a scalar by a non-scalar would cause a crash in
        type inference because TimeDeltaMixOp always assumed at least one of
        its operands was an NPTimeDelta in its generic() method.
        '''
        with self.assertRaises(TypingError) as raises:
            compile_isolated(issue_868, (types.Array(types.int32, 1, 'C'),))

        expected = (
            "Invalid use of Function(<built-in function mul>) with argument(s) of type(s): (tuple({0} x 1), {1})"
            .format(str(types.intp), types.IntegerLiteral(2)))
        self.assertIn(expected, str(raises.exception))
        self.assertIn("[1] During: typing of", str(raises.exception))

    def test_return_type_unification(self):
        with self.assertRaises(TypingError) as raises:
            compile_isolated(impossible_return_type, (types.int32,))
        self.assertIn("Can't unify return type from the following types: (), complex128",
                      str(raises.exception))

    def test_bad_hypot_usage(self):
        with self.assertRaises(TypingError) as raises:
            compile_isolated(bad_hypot_usage, ())

        errmsg = str(raises.exception)
        # Make sure it listed the known signatures.
        # This is sensitive to the formatting of the error message.
        self.assertIn(" * (float64, float64) -> float64", errmsg)

        # find the context lines
        ctx_lines = [x for x in errmsg.splitlines() if "] During" in x ]

        # Check contextual msg
        self.assertTrue(re.search(r'\[1\] During: resolving callee type: Function.*hypot', ctx_lines[0]))
        self.assertTrue(re.search(r'\[2\] During: typing of call .*test_typingerror.py', ctx_lines[1]))


    def test_imprecise_list(self):
        """
        Type inference should catch that a list type's remain imprecise,
        instead of letting lowering fail.
        """
        with self.assertRaises(TypingError) as raises:
            compile_isolated(imprecise_list, ())

        errmsg = str(raises.exception)
        msg = ("Cannot infer the type of variable 'l', have imprecise type: "
               "list(undefined)")
        self.assertIn(msg, errmsg)
        # check help message has gone in
        self.assertIn("For Numba to be able to compile a list", errmsg)

    def test_using_imprecise_list(self):
        """
        Type inference should report informative error about untyped list.
        TODO: #2931
        """
        with self.assertRaises(TypingError) as raises:
            compile_isolated(using_imprecise_list, ())

        errmsg = str(raises.exception)
        self.assertIn("Undecided type $0.6 := <undecided>", errmsg)

    def test_array_setitem_invalid_cast(self):
        with self.assertRaises(TypingError) as raises:
            compile_isolated(array_setitem_invalid_cast, ())

        errmsg = str(raises.exception)
        self.assertIn(
            "Invalid use of Function({})".format(operator.setitem),
            errmsg,
        )
        self.assertIn(
            "(array(float64, 1d, C), Literal[int](0), complex128)",
            errmsg,
        )

    def test_template_rejection_error_message_cascade(self):
        from numba import njit
        @njit
        def foo():
            z = 1
            for a, b in enumerate(z):
                pass
            return z

        with self.assertRaises(TypingError) as raises:
            foo()
        errmsg = str(raises.exception)
        expected = "All templates rejected with%s literals."
        for x in ['', 'out']:
            self.assertIn(expected % x, errmsg)

        ctx_lines = [x for x in errmsg.splitlines() if "] During" in x ]
        search = [r'\[1\] During: resolving callee type: Function.*enumerate',
                  r'\[2\] During: typing of call .*test_typingerror.py']
        for i, x in enumerate(search):
            self.assertTrue(re.search(x, ctx_lines[i]))


class TestArgumentTypingError(unittest.TestCase):
    """
    Test diagnostics of typing errors caused by argument inference failure.
    """

    def test_unsupported_array_dtype(self):
        # See issue #1943
        cfunc = jit(nopython=True)(nop)
        a = np.ones(3)
        a = a.astype(a.dtype.newbyteorder())
        with self.assertRaises(TypingError) as raises:
            cfunc(1, a, a)
        expected = textwrap.dedent("""\
            This error may have been caused by the following argument(s):
            - argument 1: Unsupported array dtype: {0}
            - argument 2: Unsupported array dtype: {0}"""
            ).format(a.dtype)
        self.assertIn(expected, str(raises.exception))

    def test_unsupported_type(self):
        cfunc = jit(nopython=True)(nop)
        foo = Foo()
        with self.assertRaises(TypingError) as raises:
            cfunc(1, foo, 1)

        expected=re.compile(("This error may have been caused by the following "
                             "argument\(s\):\\n- argument 1:.*cannot determine "
                             "Numba type of "
                             "<class \'numba.tests.test_typingerror.Foo\'>"))
        self.assertTrue(expected.search(str(raises.exception)) is not None)


class TestCallError(unittest.TestCase):
    def test_readonly_array(self):
        @jit("(f8[:],)", nopython=True)
        def inner(x):
            return x

        @jit(nopython=True)
        def outer():
            return inner(gvalues)

        gvalues = np.ones(10, dtype=np.float64)

        with self.assertRaises(TypingError) as raises:
            outer()

        got = str(raises.exception)
        pat = r"Invalid use of.*readonly array\(float64, 1d, C\)"
        self.assertIsNotNone(re.search(pat, got))


if __name__ == '__main__':
    unittest.main()
