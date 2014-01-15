from __future__ import print_function
import numba.unittest_support as unittest
from numba import bytecode, interpreter
from . import usecases


def interpret(func):
    bc = bytecode.ByteCode(func=func)
    print(bc.dump())

    interp = interpreter.Interpreter(bytecode=bc)
    interp.interpret()
    interp.dump()

    for syn in interp.syntax_info:
        print(syn)

    interp.verify()
    return interp


class TestLoopDetection(unittest.TestCase):
    def test_sum1d(self):
        interp = interpret(usecases.sum1d)
        self.assertTrue(len(interp.syntax_info) == 1)

    def test_sum2d(self):
        interp = interpret(usecases.sum2d)
        self.assertTrue(len(interp.syntax_info) == 2)

    def test_while_count(self):
        interp = interpret(usecases.while_count)
        self.assertTrue(len(interp.syntax_info) == 1)

    def test_copy_arrays(self):
        interp = interpret(usecases.copy_arrays)
        self.assertTrue(len(interp.syntax_info) == 1)

    def test_andor(self):
        interp = interpret(usecases.andor)
        self.assertTrue(len(interp.syntax_info) == 0)


if __name__ == '__main__':
    unittest.main()
