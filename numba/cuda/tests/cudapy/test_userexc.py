from __future__ import print_function, absolute_import, division
from numba.cuda.testing import unittest
from numba import cuda


class MyError(Exception):
    pass


class TestUserExc(unittest.TestCase):
    def test_user_exception(self):
        @cuda.jit("void(int32)", debug=True)
        def test_exc(x):
            if x:
                raise MyError

        test_exc(0)    # no raise
        try:
            test_exc(1)
        except MyError as e:
            print(e)

if __name__ == '__main__':
    unittest.main()

