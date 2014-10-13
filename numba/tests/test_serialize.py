from __future__ import print_function, absolute_import, division

import pickle
import sys

from numba import unittest_support as unittest
from numba import jit, types
from numba.targets import registry
from numba.typeinfer import TypingError
from .support import TestCase


@jit((types.int32, types.int32))
def add_with_sig(a, b):
    return a + b

@jit
def add_without_sig(a, b):
    return a + b

@jit(nopython=True)
def add_nopython(a, b):
    return a + b

@jit(nopython=True)
def add_nopython_fail(a, b):
    print(a.__class__)
    return a + b

def closure(a):
    @jit(nopython=True)
    def inner(b, c):
        return a + b + c
    return inner


class TestDispatcherPickling(TestCase):

    # TODO check that unpickling works from an independent process

    def run_with_protocols(self, meth, *args, **kwargs):
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            meth(proto, *args, **kwargs)

    def simulate_fresh_target(self):
        dispatcher_cls = registry.target_registry['cpu']
        # Simulate fresh targetdescr
        dispatcher_cls.targetdescr = type(dispatcher_cls.targetdescr)()

    def check_call(self, proto, func, expected_result, args):
        def check_result(func):
            if (isinstance(expected_result, type)
                and issubclass(expected_result, Exception)):
                self.assertRaises(expected_result, func, *args)
            else:
                self.assertPreciseEqual(func(*args), expected_result)

        # Control
        check_result(func)
        pickled = pickle.dumps(func, proto)
        self.simulate_fresh_target()
        new_func = pickle.loads(pickled)
        check_result(new_func)

    def test_call_with_sig(self):
        self.run_with_protocols(self.check_call, add_with_sig, 5, (1, 4))
        # Compilation has been disabled => float inputs will be coerced to int
        self.run_with_protocols(self.check_call, add_with_sig, 5, (1.2, 4.2))

    def test_call_without_sig(self):
        self.run_with_protocols(self.check_call, add_without_sig, 5, (1, 4))
        self.run_with_protocols(self.check_call, add_without_sig, 5.5, (1.2, 4.3))
        # Object mode is enabled
        self.run_with_protocols(self.check_call, add_without_sig, "abc", ("a", "bc"))

    def test_call_nopython(self):
        self.run_with_protocols(self.check_call, add_nopython, 5.5, (1.2, 4.3))
        # Object mode is disabled
        self.run_with_protocols(self.check_call, add_nopython, TypingError, ("a", "bc"))

    def test_call_nopython_fail(self):
        # Compilation fails
        self.run_with_protocols(self.check_call, add_nopython_fail, TypingError, (1, 2))

    def test_call_closure(self):
        inner = closure(1)
        self.run_with_protocols(self.check_call, inner, 6, (2, 3))


if __name__ == '__main__':
    unittest.main()
