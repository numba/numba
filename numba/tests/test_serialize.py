from __future__ import print_function, absolute_import, division

import math
import pickle
import subprocess
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

K = 3.0

from math import sqrt

def closure_with_globals(x):
    @jit(nopython=True)
    def inner(y):
        # Exercise a builtin function and a module-level constant
        k = max(K, K + 1)
        # Exercise two functions from another module, one accessed with
        # dotted notation, one imported explicitly.
        return math.hypot(x, y) + sqrt(k)
    return inner

@jit(nopython=True)
def other_function(x, y):
    return math.hypot(x, y)

def closure_calling_other_function(x):
    @jit(nopython=True)
    def inner(y, z):
        return other_function(x, y) + z
    return inner

def closure_calling_other_closure(x):
    @jit(nopython=True)
    def other_inner(y):
        return math.hypot(x, y)

    @jit(nopython=True)
    def inner(y):
        return other_inner(y) + x
    return inner


class TestDispatcherPickling(TestCase):

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

    def test_call_closure_with_globals(self):
        inner = closure_with_globals(3.0)
        self.run_with_protocols(self.check_call, inner, 7.0, (4.0,))

    def test_call_closure_calling_other_function(self):
        inner = closure_calling_other_function(3.0)
        self.run_with_protocols(self.check_call, inner, 11.0, (4.0, 6.0))

    def test_call_closure_calling_other_closure(self):
        inner = closure_calling_other_closure(3.0)
        self.run_with_protocols(self.check_call, inner, 8.0, (4.0,))

    def test_other_process(self):
        """
        Check that reconstructing doesn't depend on resources already
        instantiated in the original process.
        """
        func = closure_calling_other_closure(3.0)
        pickled = pickle.dumps(func)
        code = """if 1:
            import pickle

            data = {pickled!r}
            func = pickle.loads(data)
            res = func(4.0)
            assert res == 8.0, res
            """.format(**locals())
        subprocess.check_call([sys.executable, "-c", code])


if __name__ == '__main__':
    unittest.main()
