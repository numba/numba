from __future__ import print_function, absolute_import
import numba.unittest_support as unittest
from numba.compiler import compile_isolated
from numba import types


def return_double_or_none(x):
    if x:
        ret = None
    else:
        ret = 1.2
    return ret


def return_different_statment(x):
    if x:
        return None
    else:
        return 1.2


class TestOptional(unittest.TestCase):
    def test_return_double_or_none(self):
        pyfunc = return_double_or_none
        cres = compile_isolated(pyfunc, [types.boolean])
        cfunc = cres.entry_point

        for v in [True, False]:
            self.assertEqual(pyfunc(v), cfunc(v))

    def test_return_different_statment(self):
        pyfunc = return_different_statment
        cres = compile_isolated(pyfunc, [types.boolean])
        cfunc = cres.entry_point

        for v in [True, False]:
            self.assertEqual(pyfunc(v), cfunc(v))


# TODO: https://github.com/numba/numba/blob/double_or_none/numba/tests/test_float_or_none.py


if __name__ == '__main__':
    unittest.main()
