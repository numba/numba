from __future__ import print_function

import re

from numba.compiler import compile_isolated
from .support import TestCase
import numba.unittest_support as unittest
from numba import testing


def del_ref_func(x):
    del x
    return x


class TestLists(TestCase):

    @testing.allow_interpreter_mode
    def test_del_ref_func(self):
        pyfunc = del_ref_func
        cr = compile_isolated(pyfunc, ())
        cfunc = cr.entry_point

        errmsg = "local variable 'x' referenced before assignment"
        with self.assertRaises(UnboundLocalError) as raised:
            pyfunc(1)

        if re.search(str(raised.exception), errmsg) is None:
            self.fail("unexpected exception: {0}".format(raised.exception))

        with self.assertRaises(UnboundLocalError) as raised:
            cfunc(1)

        if re.search(str(raised.exception), errmsg) is None:
            self.fail("unexpected exception: {0}".format(raised.exception))


if __name__ == '__main__':
    unittest.main()
