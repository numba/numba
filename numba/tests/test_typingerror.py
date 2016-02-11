from __future__ import print_function

import numba.unittest_support as unittest
from numba.compiler import compile_isolated
from numba import types
from numba.errors import TypingError
from .support import TestCase

import math


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

def unknown_module():
    return numpyz.int32(0)


class TestTypingError(unittest.TestCase):

    def test_unknown_function(self):
        try:
            compile_isolated(foo, ())
        except TypingError as e:
            self.assertTrue(e.msg.startswith("Untyped global name"), e.msg)
        else:
            self.fail("Should raise error")

    def test_unknown_attrs(self):
        try:
            compile_isolated(bar, (types.int32,))
        except TypingError as e:
            self.assertTrue(e.msg.startswith("Unknown attribute"), e.msg)
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
        try:
            compile_isolated(issue_868, (types.Array(types.int32, 1, 'C'),))
        except TypingError as e:
            self.assertTrue(e.msg.startswith('Invalid usage of * '))
        else:
            self.fail('Should raise error')

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

    def test_imprecise_list(self):
        """
        Type inference should catch that a list type's remain imprecise,
        instead of letting lowering fail.
        """
        with self.assertRaises(TypingError) as raises:
            compile_isolated(imprecise_list, ())

        errmsg = str(raises.exception)
        self.assertIn("Can't infer type of variable 'l': list(undefined)",
                      errmsg)


if __name__ == '__main__':
    unittest.main()
