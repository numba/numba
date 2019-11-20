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

        @njit
        def udt():
            try:
                try:
                    print("A")
                    inner(1)
                    print("B")
                except:         # noqa: E722
                    print("C")
                    inner(2)
                    print("D")
            except:             # noqa: E722
                print("E")
                inner(3)

        with self.assertRaises(MyError) as raises:
            with captured_stdout() as stdout:
                udt()
        self.assertEqual(
            stdout.getvalue().split(),
            ["A", "call_one", "C", "call_two", "E", "call_three"],
        )
        self.assertEqual(str(raises.exception), "three")


if __name__ == '__main__':
    unittest.main()
