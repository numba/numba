from __future__ import print_function

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


if __name__ == '__main__':
    unittest.main()
