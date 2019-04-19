from __future__ import print_function

from numba import njit
import numba.unittest_support as unittest
from .support import TestCase, force_pyobj_flags



def build_map():
    return {0: 1, 2: 3}

def build_map_from_local_vars():
    # There used to be a crash due to wrong IR generation for STORE_MAP
    x = TestCase
    return {0: x, x: 1}


class DictTestCase(TestCase):

    def test_build_map(self, flags=force_pyobj_flags):
        self.run_nullary_func(build_map, flags=flags)

    def test_build_map_from_local_vars(self, flags=force_pyobj_flags):
        self.run_nullary_func(build_map_from_local_vars, flags=flags)

# XXX: requires this import sideeffect
import numba.typed


class TestCompiledDict(TestCase):
    """Testing `dict()` and `{}` usage that are redirected to
    `numba.typed.Dict`.
    """
    def test_use_dict(self):
        @njit
        def foo():
            d = dict()
            d[1] = 2
            return d

        d = foo()
        self.assertEqual(d, {1: 2})

    def test_use_curlybraces(self):
        @njit
        def foo():
            d = {}
            d[1] = 2
            return d

        d = foo()
        self.assertEqual(d, {1: 2})


    def test_use_curlybraces_with_init1(self):
        @njit
        def foo():
            return {1: 2}

        d = foo()
        self.assertEqual(d, {1: 2})

    def test_use_curlybraces_with_initmany(self):
        @njit
        def foo():
            return {1: 2.2, 3: 4.4, 5: 6.6}

        d = foo()
        self.assertEqual(d, {1: 2.2, 3: 4.4, 5: 6.6})


if __name__ == '__main__':
    unittest.main()
