import unittest

import numba
from numba import jit
from numba.core.targetconfig import Option

from numba.core.compiler import Flags
from numba.typed.typedlist import List


class MyFlags(Flags):
    extra_option = Option(
        type=bool,
        default=False,
        doc="define extra option for further extension",
    )


class MyTestCase(unittest.TestCase):

    def test_define_flag_class(self):
        my_flags = MyFlags()

        # extremely strange. If I put empty_list outside JIT, segfaults.
        # l = List.empty_list(numba.int64)

        @jit(flags=my_flags, debug=True)
        def foo():
            # put it here, don't segfault.
            l = List.empty_list(numba.int64)
            l.append(0)
            return 1

        ret = foo()
        self.assertEqual(ret, 1)


if __name__ == '__main__':
    unittest.main()
