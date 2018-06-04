from __future__ import print_function, division, absolute_import

from numba import unittest_support as unittest
from numba.transforms import find_setupwiths, with_lifting, ByPassContext
from numba.compiler import BasePipeline
from numba.bytecode import FunctionIdentity, ByteCode
from numba.interpreter import Interpreter
from .support import TestCase, tag, MemoryLeakMixin, captured_stdout


def get_func_ir(func):
    func_id = FunctionIdentity.from_function(func)
    bc = ByteCode(func_id=func_id)
    interp = Interpreter(func_id)
    func_ir = interp.interpret(bc)
    return func_ir


def lift1():
    print("A")
    with ByPassContext:
        print("B")
        b()
    print("C")


def lift2():
    print("A")
    with ByPassContext:
        print("B")
        b()
    with ByPassContext:
        print("C")
        b()
    print("D")


def lift3():
    print("A")
    with ByPassContext:
        print("B")
        b()
        with ByPassContext:
            print("C")
            b()
    print("D")


def lift4():
    print("A")
    with ByPassContext:
        print("B")
        b()
        with ByPassContext:
            print("C")
            b()
    with ByPassContext:
        print("D")
        b()
    print("E")


def lift5():
    print("A")


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


class TestWithLifting(TestCase):
    def check_extracted_with(self, func, expect_count, expected_stdout):
        # import dis

        # dis.dis(func)
        # print("++++--")
        the_ir = get_func_ir(func)
        new_ir, extracted = with_lifting(the_ir)
        self.assertEqual(len(extracted), expect_count)
        # print(new_ir.dump())
        # print("+++")
        # print(the_ir.dump())
        cres = self.compile_ir(new_ir)

        with captured_stdout() as out:
            cres.entry_point()

        self.assertEqual(out.getvalue(), expected_stdout)

    def compile_ir(self, the_ir, args=(), return_type=None):
        from numba import typing
        from numba.targets.registry import cpu_target
        from numba.targets import cpu
        from numba.compiler import compile_ir, DEFAULT_FLAGS
        typingctx = typing.Context()
        targetctx = cpu.CPUContext(typingctx)
        flags = DEFAULT_FLAGS
        # Register the contexts in case for nested @jit or @overload calls
        with cpu_target.nested_context(typingctx, targetctx):
            return compile_ir(typingctx, targetctx, the_ir, args,
                              return_type, flags, locals={})

    def test_lift1(self):
        self.check_extracted_with(lift1, expect_count=1,
                                  expected_stdout="A\nC\n")

    def test_lift2(self):
        self.check_extracted_with(lift2, expect_count=2,
                                  expected_stdout="A\nD\n")

    def test_lift3(self):
        self.check_extracted_with(lift3, expect_count=1,
                                  expected_stdout="A\nD\n")

    def test_lift4(self):
        self.check_extracted_with(lift4, expect_count=2,
                                  expected_stdout="A\nE\n")

    def test_lift5(self):
        self.check_extracted_with(lift5, expect_count=0,
                                  expected_stdout="A\n")


if __name__ == '__main__':
    unittest.main()
