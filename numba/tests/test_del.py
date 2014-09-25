from __future__ import print_function
from numba.compiler import compile_isolated
from numba.tests.support import TestCase
import numba.unittest_support as unittest
from numba import testing


def del_list_item_func(x):
    del x[0]
    return x


def del_ref_func(x):
    del x
    return x


class TestLists(TestCase):

    @testing.allow_interpreter_mode
    def test_del_list_item_func(self):
        pyfunc = del_list_item_func
        cr = compile_isolated(pyfunc, ())
        cfunc = cr.entry_point
        expected = pyfunc([1, 2, 3])
        result = cfunc([1, 2, 3])
        self.assertEqual(expected, result)

    @testing.allow_interpreter_mode
    def test_del_ref_func(self):
        pyfunc = del_ref_func
        cr = compile_isolated(pyfunc, ())
        cfunc = cr.entry_point

        errmsg = "local variable 'x' referenced before assignment"
        with self.assertRaisesRegex(UnboundLocalError, errmsg):
            pyfunc(1)

        with self.assertRaisesRegex(UnboundLocalError, errmsg):
            cfunc(1)


if __name__ == '__main__':
    unittest.main()
