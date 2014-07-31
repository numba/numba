
from __future__ import print_function, absolute_import

import gc
import weakref

from numba import unittest_support as unittest
from numba.utils import IS_PY3
from numba import jit, types
from .support import TestCase


class Dummy(object):

    def __add__(self, other):
        return other + 5


def global_usecase1(x):
    return x + 1

def global_usecase2():
    return global_obj + 1


class TestFuncLifetime(TestCase):
    """
    Test the lifetime of compiled function objects and their dependencies.
    """
    # NOTE: there's a test for closure lifetime in test_closure

    def get_impl(self, dispatcher):
        """
        Get the single implementation (a C function object) of a dispatcher.
        """
        self.assertEqual(len(dispatcher.overloads), 1)
        return list(dispatcher.overloads.values())[0]

    def check_local_func_lifetime(self, **jitargs):
        def f(x):
            return x + 1

        c_f = jit('int32(int32)', **jitargs)(f)
        self.assertPreciseEqual(c_f(1), 2)

        cfunc = self.get_impl(c_f)

        # Since we can't take a weakref to a C function object
        # (see http://bugs.python.org/issue22116), ensure it's
        # collected by taking a weakref to its __self__ instead
        # (a _dynfunc._Closure object).
        refs = [weakref.ref(obj) for obj in (f, c_f, cfunc.__self__)]
        obj = f = c_f = cfunc = None
        gc.collect()
        self.assertEqual([wr() for wr in refs], [None] * len(refs))

    def test_local_func_lifetime(self):
        self.check_local_func_lifetime(forceobj=True)

    def test_local_func_lifetime_npm(self):
        self.check_local_func_lifetime(nopython=True)

    def check_global_func_lifetime(self, **jitargs):
        c_f = jit(**jitargs)(global_usecase1)
        self.assertPreciseEqual(c_f(1), 2)

        cfunc = self.get_impl(c_f)

        wr = weakref.ref(c_f)
        refs = [weakref.ref(obj) for obj in (c_f, cfunc.__self__)]
        obj = c_f = cfunc = None
        gc.collect()
        self.assertEqual([wr() for wr in refs], [None] * len(refs))

    def test_global_func_lifetime(self):
        self.check_global_func_lifetime(forceobj=True)

    def test_global_func_lifetime_npm(self):
        self.check_global_func_lifetime(nopython=True)

    def check_global_obj_lifetime(self, **jitargs):
        # Since global objects can be recorded for typing purposes,
        # check that they are not kept around after they are removed
        # from the globals.
        global global_obj
        global_obj = Dummy()

        c_f = jit(**jitargs)(global_usecase2)
        self.assertPreciseEqual(c_f(), 6)

        # XXX global_obj will survive, perhaps because of buggy code
        # generation.
        refs = [weakref.ref(obj) for obj in (c_f, global_obj)]
        obj = c_f = global_obj = None
        gc.collect()
        self.assertEqual([wr() for wr in refs], [None] * len(refs))

    @unittest.expectedFailure
    def test_global_obj_lifetime(self):
        self.check_global_obj_lifetime(forceobj=True)



if __name__ == '__main__':
    unittest.main()
