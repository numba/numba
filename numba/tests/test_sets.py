from __future__ import print_function

import numba.unittest_support as unittest
from numba.utils import PYVERSION
from .support import TestCase, enable_pyobj_flags


def build_set_usecase(*args):
    ns = {}
    src = """if 1:
    def build_set():
        return {%s}
    """ % ', '.join(repr(arg) for arg in args)
    code = compile(src, '<>', 'exec')
    eval(code, ns)
    return ns['build_set']


needs_set_literals = unittest.skipIf(PYVERSION < (2, 7),
                                     "set literals unavailable before Python 2.7")


class SetTestCase(TestCase):

    @needs_set_literals
    def test_build_set(self, flags=enable_pyobj_flags):
        pyfunc = build_set_usecase(1, 2, 3, 2)
        self.run_nullary_func(pyfunc, flags=flags)

    @needs_set_literals
    def test_build_heterogenous_set(self, flags=enable_pyobj_flags):
        pyfunc = build_set_usecase(1, 2.0, 3j, 2)
        self.run_nullary_func(pyfunc, flags=flags)
        # Check that items are inserted in the right order (here the
        # result will be {2}, not {2.0})
        pyfunc = build_set_usecase(2.0, 2)
        got, expected = self.run_nullary_func(pyfunc, flags=flags)
        self.assertIs(type(got.pop()), type(expected.pop()))


if __name__ == '__main__':
    unittest.main()
