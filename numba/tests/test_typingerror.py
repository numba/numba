from __future__ import print_function
import numba.unittest_support as unittest
from numba.compiler import compile_isolated
from numba import types
from numba.typeinfer import TypingError


def what():
    pass


def foo():
    return what()


def bar(x):
    return x.a


class TestTypingError(unittest.TestCase):
    def test_unknown_function(self):
        try:
            compile_isolated(foo, ())
        except TypingError as e:
            self.assertTrue(e.msg.startswith("Untyped global name"))
        else:
            self.fail("Should raise error")

    def test_unknown_attrs(self):
        try:
            compile_isolated(bar, (types.int32,))
        except TypingError as e:
            self.assertTrue(e.msg.startswith("Unknown attribute"))
        else:
            self.fail("Should raise error")


if __name__ == '__main__':
    unittest.main()
