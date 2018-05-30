from __future__ import print_function, division, absolute_import

from numba import unittest_support as unittest
from numba.transforms import find_setupwiths, with_lifting
from numba.compiler import BasePipeline
from numba.bytecode import FunctionIdentity, ByteCode
from numba.interpreter import Interpreter
from .support import TestCase, tag, MemoryLeakMixin


def get_func_ir(func):
    func_id = FunctionIdentity.from_function(func)
    bc = ByteCode(func_id=func_id)
    interp = Interpreter(func_id)
    func_ir = interp.interpret(bc)
    return func_ir


def lift1():
    with a():
        b()


def lift2():
    with a():
        b()
    with a():
        b()


def lift3():
    with a():
        b()
        with a():
            b()


def lift4():
    with a():
        b()
        with a():
            b()
    with a():
        b()


def lift5():
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


class TestWithLifting(TestCase):
    def check_extracted_with(self, func, expect_count):
        the_ir = get_func_ir(func)
        new_ir, extracted = with_lifting(the_ir)
        self.assertEqual(len(extracted), expect_count)

    def test_lift1(self):
        self.check_extracted_with(lift1, expect_count=1)

    def test_lift2(self):
        self.check_extracted_with(lift2, expect_count=2)

    def test_lift3(self):
        self.check_extracted_with(lift3, expect_count=1)

    def test_lift4(self):
        self.check_extracted_with(lift4, expect_count=2)

    def test_lift5(self):
        self.check_extracted_with(lift5, expect_count=0)


if __name__ == '__main__':
    unittest.main()
