import unittest

from numba import jit
from numba.core.targetconfig import Option

from numba.core.compiler import Flags


class MyFlags(Flags):
    extra_option = Option(
        type=bool,
        default=False,
        doc="define extra option for further extension",
    )


class MyTestCase(unittest.TestCase):

    def test_define_flag_class(self):
        # @jit(flag_class=MyFlags)
        @jit()
        def foo():
            return 1

        ret = foo()
        self.assertEqual(ret, 1)


if __name__ == '__main__':
    unittest.main()
