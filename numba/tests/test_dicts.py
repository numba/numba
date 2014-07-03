from __future__ import print_function

import numba.unittest_support as unittest
from .support import TestCase, force_pyobj_flags


def build_map():
    return {0: 1, 2: 3}


class DictTestCase(TestCase):

    def test_build_map(self, flags=force_pyobj_flags):
        self.run_nullary_func(build_map, flags=flags)


if __name__ == '__main__':
    unittest.main()
