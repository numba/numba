from __future__ import print_function

from itertools import product

from numba import njit, typed, objmode
from numba.errors import UnsupportedError, CompilerError
from .support import (
    TestCase, unittest, captured_stdout, skip_tryexcept_unsupported,
    skip_tryexcept_supported, MemoryLeakMixin
)


class MyError(Exception):
    pass


@skip_tryexcept_unsupported
class TestTryBareExcept(TestCase):
    """Test the following pattern:

        try:
            <body>
        except:
            <handling>
    """
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

    def _multi_inner(self):
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

        return inner

    def test_nested_try(self):
        inner = self._multi_inner()

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

    def test_loop_in_try(self):
        inner = self._multi_inner()

        @njit
        def udt(x, n):
            try:
                print("A")
                for i in range(n):
                    print(i)
                    if i == x:
                        inner(i)
            except:             # noqa: E722
                print("B")
            return i

        # case 1
        with captured_stdout() as stdout:
            res = udt(3, 5)
        self.assertEqual(
            stdout.getvalue().split(),
            ["A", "0", "1", "2", "3", "call_three", "B"],
        )
        self.assertEqual(res, 3)

        # case 2
        with captured_stdout() as stdout:
            res = udt(1, 3)
        self.assertEqual(
            stdout.getvalue().split(),
            ["A", "0", "1", "call_one", "B"],
        )
        self.assertEqual(res, 1)

        # case 3
        with captured_stdout() as stdout:
            res = udt(0, 3)
        self.assertEqual(
            stdout.getvalue().split(),
            ["A", "0", "call_other", "1", "2"],
        )
        self.assertEqual(res, 2)

    def test_raise_in_try(self):
        @njit
        def udt(x):
            try:
                print("A")
                if x:
                    raise MyError("my_error")
                print("B")
            except:             # noqa: E722
                print("C")
                return 321
            return 123

        # case 1
        with captured_stdout() as stdout:
            res = udt(True)

        self.assertEqual(
            stdout.getvalue().split(),
            ["A", "C"],
        )
        self.assertEqual(res, 321)

        # case 2
        with captured_stdout() as stdout:
            res = udt(False)

        self.assertEqual(
            stdout.getvalue().split(),
            ["A", "B"],
        )
        self.assertEqual(res, 123)


@skip_tryexcept_unsupported
class TestTryExceptCaught(TestCase):
    def test_catch_exception(self):
        @njit
        def udt(x):
            try:
                print("A")
                if x:
                    raise ZeroDivisionError("321")
                print("B")
            except Exception:
                print("C")
            print("D")

        # case 1
        with captured_stdout() as stdout:
            udt(True)

        self.assertEqual(
            stdout.getvalue().split(),
            ["A", "C", "D"],
        )

        # case 2
        with captured_stdout() as stdout:
            udt(False)

        self.assertEqual(
            stdout.getvalue().split(),
            ["A", "B", "D"],
        )

    def test_return_in_catch(self):
        @njit
        def udt(x):
            try:
                print("A")
                if x:
                    raise ZeroDivisionError
                print("B")
                r = 123
            except Exception:
                print("C")
                r = 321
                return r
            print("D")
            return r

        # case 1
        with captured_stdout() as stdout:
            res = udt(True)

        self.assertEqual(
            stdout.getvalue().split(),
            ["A", "C"],
        )
        self.assertEqual(res, 321)

        # case 2
        with captured_stdout() as stdout:
            res = udt(False)

        self.assertEqual(
            stdout.getvalue().split(),
            ["A", "B", "D"],
        )
        self.assertEqual(res, 123)

    def test_save_caught(self):
        @njit
        def udt(x):
            try:
                print("A")
                if x:
                    raise ZeroDivisionError
                print("B")
                r = 123
            except Exception as e:  # noqa: F841
                print("C")
                r = 321
                return r
            print("D")
            return r

        # case 1
        with captured_stdout() as stdout:
            res = udt(True)

        self.assertEqual(
            stdout.getvalue().split(),
            ["A", "C"],
        )
        self.assertEqual(res, 321)

        # case 2
        with captured_stdout() as stdout:
            res = udt(False)

        self.assertEqual(
            stdout.getvalue().split(),
            ["A", "B", "D"],
        )
        self.assertEqual(res, 123)


@skip_tryexcept_unsupported
class TestTryExceptNested(TestCase):
    "Tests for complicated nesting"

    def check_compare(self, cfunc, pyfunc, *args, **kwargs):
        with captured_stdout() as stdout:
            pyfunc(*args, **kwargs)
        expect = stdout.getvalue()

        with captured_stdout() as stdout:
            cfunc(*args, **kwargs)
        got = stdout.getvalue()
        self.assertEqual(
            expect, got,
            msg="args={} kwargs={}".format(args, kwargs)
        )

    def test_try_except_else(self):
        @njit
        def udt(x, y, z, p):
            print('A')
            if x:
                print('B')
                try:
                    print('C')
                    if y:
                        print('D')
                        raise MyError("y")
                    print('E')
                except Exception as e: # noqa: F841
                    print('F')
                    try:
                        print('H')
                        try:
                            print('I')
                            if z:
                                print('J')
                                raise MyError('z')
                            print('K')
                        except Exception:
                            print('L')
                        else:
                            print('M')
                    except Exception:
                        print('N')
                    else:
                        print('O')
                    print('P')
                else:
                    print('G')
                print('Q')
            print('R')

        cases = list(product([True, False], repeat=4))
        self.assertTrue(cases)
        for x, y, z, p in cases:
            self.check_compare(
                udt, udt.py_func,
                x=x, y=y, z=z, p=p,
            )

    def test_try_except_finally(self):
        @njit
        def udt(p, q):
            try:
                print('A')
                if p:
                    print('B')
                    raise MyError
                print('C')
            except:             # noqa: E722
                print('D')
            finally:
                try:
                    print('E')
                    if q:
                        print('F')
                        raise MyError
                except Exception:
                    print('G')
                else:
                    print('H')
                finally:
                    print('I')

        cases = list(product([True, False], repeat=2))
        self.assertTrue(cases)
        for p, q in cases:
            self.check_compare(
                udt, udt.py_func,
                p=p, q=q,
            )


@skip_tryexcept_supported
class TestTryExceptUnsupported(TestCase):

    msg_pattern = "'try' block not supported until python3.7 or later"

    def check(self, call, *args):
        with self.assertRaises(UnsupportedError) as raises:
            call(*args)
        self.assertIn(self.msg_pattern, str(raises.exception))

    def test_try_except(self):
        @njit
        def foo(x):
            try:
                if x:
                    raise MyError
            except:   # noqa: E722
                pass
        self.check(foo, True)

    def test_try_finally(self):
        @njit
        def foo(x):
            try:
                if x:
                    raise MyError
            finally:
                pass
        self.check(foo, True)


class TestTryExceptRefct(MemoryLeakMixin, TestCase):
    def test_list_direct_raise(self):
        @njit
        def udt(n, raise_at):
            lst = typed.List()
            try:
                for i in range(n):
                    if i == raise_at:
                        raise IndexError
                    lst.append(i)
            except Exception:
                return lst
            else:
                return lst

        out = udt(10, raise_at=5)
        self.assertEqual(list(out), list(range(5)))
        out = udt(10, raise_at=10)
        self.assertEqual(list(out), list(range(10)))

    def test_list_indirect_raise(self):
        @njit
        def appender(lst, n, raise_at):
            for i in range(n):
                if i == raise_at:
                    raise IndexError
                lst.append(i)
            return lst

        @njit
        def udt(n, raise_at):
            lst = typed.List()
            lst.append(0xbe11)
            try:
                appender(lst, n, raise_at)
            except Exception:
                return lst
            else:
                return lst

        out = udt(10, raise_at=5)
        self.assertEqual(list(out), [0xbe11] + list(range(5)))
        out = udt(10, raise_at=10)
        self.assertEqual(list(out), [0xbe11] + list(range(10)))


class TestTryExceptOtherControlFlow(TestCase):
    def test_yield(self):
        @njit
        def udt(n, x):
            for i in range(n):
                try:
                    if i == x:
                        raise ValueError
                    yield i
                except Exception:
                    return

        self.assertEqual(list(udt(10, 5)), list(range(5)))
        self.assertEqual(list(udt(10, 10)), list(range(10)))

    def test_objmode(self):
        @njit
        def udt():
            try:
                with objmode():
                    print(object())
            except Exception:
                return

        with self.assertRaises(CompilerError) as raises:
            udt()
        self.assertIn(
            "Does not support with-context that contain branches",
            str(raises.exception),
        )


if __name__ == '__main__':
    unittest.main()
