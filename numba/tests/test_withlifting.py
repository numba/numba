from __future__ import print_function, division, absolute_import

import copy
import numpy as np

from numba import unittest_support as unittest
from numba.transforms import find_setupwiths, with_lifting
from numba.withcontexts import bypass_context, call_context, objmode_context
from numba.bytecode import FunctionIdentity, ByteCode
from numba.interpreter import Interpreter
from numba import typing, errors
from numba.targets.registry import cpu_target
from numba.targets import cpu
from numba.compiler import compile_ir, DEFAULT_FLAGS
from numba import njit, typeof
from .support import MemoryLeak, TestCase, captured_stdout


def get_func_ir(func):
    func_id = FunctionIdentity.from_function(func)
    bc = ByteCode(func_id=func_id)
    interp = Interpreter(func_id)
    func_ir = interp.interpret(bc)
    return func_ir


def lift1():
    print("A")
    with bypass_context:
        print("B")
        b()
    print("C")


def lift2():
    x = 1
    print("A", x)
    x = 1
    with bypass_context:
        print("B", x)
        x += 100
        b()
    x += 1
    with bypass_context:
        print("C", x)
        b()
        x += 10
    x += 1
    print("D", x)


def lift3():
    x = 1
    y = 100
    print("A", x, y)
    with bypass_context:
        print("B")
        b()
        x += 100
        with bypass_context:
            print("C")
            y += 100000
            b()
    x += 1
    y += 1
    print("D", x, y)


def lift4():
    x = 0
    print("A", x)
    x += 10
    with bypass_context:
        print("B")
        b()
        x += 1
        for i in range(10):
            with bypass_context:
                print("C")
                b()
                x += i
    with bypass_context:
        print("D")
        b()
        if x:
            x *= 10
    x += 1
    print("E", x)


def lift5():
    print("A")


def liftcall1():
    x = 1
    print("A", x)
    with call_context:
        x += 1
    print("B", x)
    return x


def liftcall2():
    x = 1
    print("A", x)
    with call_context:
        x += 1
    print("B", x)
    with call_context:
        x += 10
    print("C", x)
    return x


def liftcall3():
    x = 1
    print("A", x)
    with call_context:
        if x > 0:
            x += 1
    print("B", x)
    with call_context:
        for i in range(10):
            x += i
    print("C", x)
    return x


def liftcall4():
    with call_context:
        with call_context:
            pass


def lift_undefiend():
    with undefined_global_var:
        pass


bogus_contextmanager = object()


def lift_invalid():
    with bogus_contextmanager:
        pass


class TestWithFinding(TestCase):
    def check_num_of_with(self, func, expect_count):
        the_ir = get_func_ir(func)
        ct = len(find_setupwiths(the_ir.blocks))
        self.assertEqual(ct, expect_count)

    def test_lift1(self):
        self.check_num_of_with(lift1, expect_count=1)

    def test_lift2(self):
        self.check_num_of_with(lift2, expect_count=2)

    def test_lift3(self):
        self.check_num_of_with(lift3, expect_count=1)

    def test_lift4(self):
        self.check_num_of_with(lift4, expect_count=2)

    def test_lift5(self):
        self.check_num_of_with(lift5, expect_count=0)


class BaseTestWithLifting(TestCase):
    def setUp(self):
        super(BaseTestWithLifting, self).setUp()
        self.typingctx = typing.Context()
        self.targetctx = cpu.CPUContext(self.typingctx)
        self.flags = DEFAULT_FLAGS

    def check_extracted_with(self, func, expect_count, expected_stdout):
        the_ir = get_func_ir(func)
        new_ir, extracted = with_lifting(
            the_ir, self.typingctx, self.targetctx, self.flags,
            locals={},
        )
        self.assertEqual(len(extracted), expect_count)
        cres = self.compile_ir(new_ir)

        with captured_stdout() as out:
            cres.entry_point()

        self.assertEqual(out.getvalue(), expected_stdout)

    def compile_ir(self, the_ir, args=(), return_type=None):
        typingctx = self.typingctx
        targetctx = self.targetctx
        flags = self.flags
        # Register the contexts in case for nested @jit or @overload calls
        with cpu_target.nested_context(typingctx, targetctx):
            return compile_ir(typingctx, targetctx, the_ir, args,
                              return_type, flags, locals={})


class TestLiftByPass(BaseTestWithLifting):

    def test_lift1(self):
        self.check_extracted_with(lift1, expect_count=1,
                                  expected_stdout="A\nC\n")

    def test_lift2(self):
        self.check_extracted_with(lift2, expect_count=2,
                                  expected_stdout="A 1\nD 3\n")

    def test_lift3(self):
        self.check_extracted_with(lift3, expect_count=1,
                                  expected_stdout="A 1 100\nD 2 101\n")

    def test_lift4(self):
        self.check_extracted_with(lift4, expect_count=2,
                                  expected_stdout="A 0\nE 11\n")

    def test_lift5(self):
        self.check_extracted_with(lift5, expect_count=0,
                                  expected_stdout="A\n")


class TestLiftCall(BaseTestWithLifting):

    def check_same_semantic(self, func):
        """Ensure same semantic with non-jitted code
        """
        jitted = njit(func)
        with captured_stdout() as got:
            jitted()

        with captured_stdout() as expect:
            func()

        self.assertEqual(got.getvalue(), expect.getvalue())

    def test_liftcall1(self):
        self.check_extracted_with(liftcall1, expect_count=1,
                                  expected_stdout="A 1\nB 2\n")
        self.check_same_semantic(liftcall1)

    def test_liftcall2(self):
        self.check_extracted_with(liftcall2, expect_count=2,
                                  expected_stdout="A 1\nB 2\nC 12\n")
        self.check_same_semantic(liftcall2)

    def test_liftcall3(self):
        self.check_extracted_with(liftcall3, expect_count=2,
                                  expected_stdout="A 1\nB 2\nC 47\n")
        self.check_same_semantic(liftcall3)

    def test_liftcall4(self):
        with self.assertRaises(errors.TypingError) as raises:
            njit(liftcall4)()
        # Known error.  We only support one context manager per function
        # for body that are lifted.
        self.assertIn("re-entrant", str(raises.exception))


class TestLiftObj(MemoryLeak, TestCase):

    def assert_equal_return_and_stdout(self, pyfunc, *args):
        py_args = copy.deepcopy(args)
        c_args = copy.deepcopy(args)
        cfunc = njit(pyfunc)

        with captured_stdout() as stream:
            expect_res = pyfunc(*py_args)
            expect_out = stream.getvalue()

        # avoid compiling during stdout-capturing for easier print-debugging
        cfunc.compile(tuple(map(typeof, c_args)))
        with captured_stdout() as stream:
            got_res = cfunc(*c_args)
            got_out = stream.getvalue()

        self.assertEqual(expect_out, got_out)
        self.assertPreciseEqual(expect_res, got_res)

    def test_lift_objmode_basic(self):
        def bar(ival):
            print("ival =", {'ival': ival // 2})

        def foo(ival):
            ival += 1
            with objmode_context:
                bar(ival)
            return ival + 1

        self.assert_equal_return_and_stdout(foo, 123)

    def test_lift_objmode_array_in(self):
        def bar(arr):
            print({'arr': arr // 2})
            # arr is modified. the effect is visible outside.
            arr *= 2

        def foo(nelem):
            arr = np.arange(nelem)
            with objmode_context:
                # arr is modified inplace inside bar()
                bar(arr)
            return arr + 1

        nelem = 10
        self.assert_equal_return_and_stdout(foo, nelem)

    def test_lift_objmode_define_new_unused(self):
        def bar(y):
            print(y)

        def foo(x):
            with objmode_context():
                y = 2 + x           # defined but unused outside
                a = np.arange(y)    # defined but unused outside
                bar(a)
            return x

        arg = 123
        self.assert_equal_return_and_stdout(foo, arg)

    def test_lift_objmode_return_simple(self):
        def inverse(x):
            print(x)
            return 1 / x

        def foo(x):
            with objmode_context(y="float64"):
                y = inverse(x)
            return x, y

        arg = 123
        self.assert_equal_return_and_stdout(foo, arg)

    def test_lift_objmode_return_array(self):
        def inverse(x):
            print(x)
            return 1 / x

        def foo(x):
            with objmode_context(y="float64[:]", z="int64"):
                y = inverse(x)
                z = int(y[0])
            return x, y, z

        arg = np.arange(1, 10, dtype=np.float64)
        self.assert_equal_return_and_stdout(foo, arg)

    def test_lift_objmode_using_list(self):
        def foo(x):
            with objmode_context(y="float64[:]"):
                print(x)
                x[0] = 4
                print(x)
                y = [1, 2, 3] + x
                y = np.asarray([1 / i for i in y])
            return x, y

        arg = [1, 2, 3]
        self.assert_equal_return_and_stdout(foo, arg)

    def test_lift_objmode_var_redef(self):
        def foo(x):
            for x in range(x):
                pass
            if x:
                x += 1
            with objmode_context(x="intp"):
                print(x)
                x -= 1
                print(x)
                for i in range(x):
                    x += i
                    print(x)
            return x

        arg = 123
        self.assert_equal_return_and_stdout(foo, arg)


class TestBogusContext(BaseTestWithLifting):
    def test_undefined_global(self):
        the_ir = get_func_ir(lift_undefiend)

        with self.assertRaises(errors.CompilerError) as raises:
            with_lifting(
                the_ir, self.typingctx, self.targetctx, self.flags, locals={},
            )
        self.assertIn(
            "Undefined variable used as context manager",
            str(raises.exception),
            )

    def test_invalid(self):
        the_ir = get_func_ir(lift_invalid)

        with self.assertRaises(errors.CompilerError) as raises:
            with_lifting(
                the_ir, self.typingctx, self.targetctx, self.flags, locals={},
            )
        self.assertIn(
            "Unsupported context manager in use",
            str(raises.exception),
            )


if __name__ == '__main__':
    unittest.main()
