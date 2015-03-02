from __future__ import print_function, absolute_import, division

from numba.cuda.testing import unittest
from numba import cuda


class MyError(Exception):
    pass


class TestUserExc(unittest.TestCase):
    def test_user_exception(self):
        @cuda.jit("void(int32)", debug=True)
        def test_exc(x):
            if x == 1:
                raise MyError
            elif x == 2:
                raise MyError("foo")

        test_exc(0)    # no raise
        with self.assertRaises(MyError) as cm:
            test_exc(1)
        self.assertEqual("tid=[0, 0, 0] ctaid=[0, 0, 0]", str(cm.exception))
        with self.assertRaises(MyError) as cm:
            test_exc(2)
        self.assertEqual("tid=[0, 0, 0] ctaid=[0, 0, 0]: foo", str(cm.exception))


if __name__ == '__main__':
    unittest.main()

