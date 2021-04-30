from numba.cuda.testing import unittest, CUDATestCase
from numba import cuda
from numba.core import config


class MyError(Exception):
    pass


regex_pattern = (
    r'In function [\'"]test_exc[\'"], file [\:\.\/\\\-a-zA-Z_0-9]+, line \d+'
)


class TestUserExc(CUDATestCase):

    def test_user_exception(self):
        @cuda.jit("void(int32)", debug=True)
        def test_exc(x):
            if x == 1:
                raise MyError
            elif x == 2:
                raise MyError("foo")

        test_exc[1, 1](0)    # no raise
        with self.assertRaises(MyError) as cm:
            test_exc[1, 1](1)
        if not config.ENABLE_CUDASIM:
            self.assertRegexpMatches(str(cm.exception), regex_pattern)
        self.assertIn("tid=[0, 0, 0] ctaid=[0, 0, 0]", str(cm.exception))
        with self.assertRaises(MyError) as cm:
            test_exc[1, 1](2)
        if not config.ENABLE_CUDASIM:
            self.assertRegexpMatches(str(cm.exception), regex_pattern)
            self.assertRegexpMatches(str(cm.exception), regex_pattern)
        self.assertIn("tid=[0, 0, 0] ctaid=[0, 0, 0]: foo", str(cm.exception))


if __name__ == '__main__':
    unittest.main()
