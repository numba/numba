from __future__ import print_function

import numba.unittest_support as unittest
from numba.compiler import compile_isolated
from numba import types
from numba.errors import TypingError


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

    def test_issue_868(self):
        '''
        Summary: multiplying a scalar by a non-scalar would cause a crash in
        type inference because TimeDeltaMixOp always assumed at least one of
        its operands was an NPTimeDelta in its generic() method.
        '''
        try:
            compile_isolated(issue_868, (types.Array(types.int32, 1, 'C'),))
        except TypingError as e:
            self.assertTrue(e.msg.startswith('Undeclared'))
        else:
            self.fail('Should raise error')

    def test_return_type_unification(self):
        with self.assertRaises(TypingError) as raises:
            compile_isolated(impossible_return_type, (types.int32,))
        self.assertIn("Can't unify return type from the following types: (), complex128",
                      str(raises.exception))


if __name__ == '__main__':
    unittest.main()
