from __future__ import print_function, division, absolute_import

from numba import unittest_support as unittest
from numba.transforms import find_setupwiths, with_lifting
from numba.withcontexts import bypass_context, call_context
from numba.bytecode import FunctionIdentity, ByteCode
from numba.interpreter import Interpreter
from numba import typing
from numba.targets.registry import cpu_target
from numba.targets import cpu
from numba.compiler import compile_ir, DEFAULT_FLAGS
from .support import TestCase, MemoryLeakMixin, captured_stdout


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
        # print("B", x)  # pending env patch
        x += 1
        # print("C", x)  # pending env patch
    print("D", x)
    return x


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
            the_ir, self.typingctx, self.targetctx, self.flags, locals={},
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

    def check_extracted_with(self, func, expect_count, expected_stdout):
        from numba import njit

        jitted = njit(func)
        with captured_stdout() as out:
            jitted()

        self.assertEqual(out.getvalue(), expected_stdout)

    def test_liftcall1(self):
        self.check_extracted_with(liftcall1, expect_count=1,
                                  expected_stdout="A 1\nD 2\n")


if __name__ == '__main__':
    unittest.main()
