from numba import njit
from numba.typed import List
from .support import TestCase, MemoryLeakMixin


# import the symbol from _future_
from numba.future import disable_reflected_list
# make sure we can call it to avoid linter complaints
disable_reflected_list()


class TestTypedList(MemoryLeakMixin, TestCase):

    def test(self):
        @njit
        def foo():
            l = [0]
            return l

        # the JITed function will return a Numba typed-list
        cfunc_expected = List()
        cfunc_expected.append(0)
        cfunc_received = foo()
        self.assertEqual(cfunc_expected, cfunc_received)
        self.assertTrue(isinstance(cfunc_received, List))

        # the non-JITed version will return a Python list
        pyfunc_expected = [0]
        pyfunc_received = foo.py_func()
        self.assertEqual(pyfunc_expected, pyfunc_received)
        self.assertTrue(isinstance(pyfunc_received, list))
