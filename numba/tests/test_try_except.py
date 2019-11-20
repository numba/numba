from __future__ import print_function

from numba import njit
from .support import TestCase, unittest, captured_stdout


class MyError(Exception):
    pass


class TestTryExcept(TestCase):
    def test_try_inner_raise(self):
        @njit
        def inner(x):
            if x:
                raise MyError

        @njit
        def udt(x):
            try:
                inner(x)
                return "not raised"
            except:             # noqa: E722
                return "caught"

        self.assertEqual(udt(False), "not raised")
        self.assertEqual(udt(True), "caught")

    def test_try_state_reset(self):
        @njit
        def inner(x):
            if x == 1:
                raise MyError("one")
            elif x == 2:
                raise MyError("two")

        @njit
        def udt(x):
            try:
                inner(x)
                res = "not raised"
            except:             # noqa: E722
                res = "caught"
            if x == 0:
                inner(2)
            return res

        with self.assertRaises(MyError) as raises:
            udt(0)
        self.assertEqual(str(raises.exception), "two")
        self.assertEqual(udt(1), "caught")
        self.assertEqual(udt(-1), "not raised")

    def test_nested_try(self):
        @njit
        def inner(x):
            if x == 1:
                print("call_one")
                raise MyError("one")
            elif x == 2:
                print("call_two")
                raise MyError("two")
            elif x == 3:
                print("call_three")
                raise MyError("three")
            else:
                print("call_other")

        @njit
        def udt(x, y, z):
            try:
                try:
                    print("A")
                    inner(x)
                    print("B")
                except:         # noqa: E722
                    print("C")
                    inner(y)
                    print("D")
            except:             # noqa: E722
                print("E")
                inner(z)
                print("F")

        # case 1
        with self.assertRaises(MyError) as raises:
            with captured_stdout() as stdout:
                udt(1, 2, 3)
        self.assertEqual(
            stdout.getvalue().split(),
            ["A", "call_one", "C", "call_two", "E", "call_three"],
        )
        self.assertEqual(str(raises.exception), "three")

        # case 2
        with captured_stdout() as stdout:
            udt(1, 0, 3)
        self.assertEqual(
            stdout.getvalue().split(),
            ["A", "call_one", "C", "call_other", "D"],
        )

        # case 3
        with captured_stdout() as stdout:
            udt(1, 2, 0)
        self.assertEqual(
            stdout.getvalue().split(),
            ["A", "call_one", "C", "call_two", "E", "call_other", "F"],
        )


if __name__ == '__main__':
    unittest.main()
