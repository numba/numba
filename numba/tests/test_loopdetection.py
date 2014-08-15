from __future__ import print_function

import numba.unittest_support as unittest
from numba import bytecode, interpreter, utils
from . import usecases


def interpret(func):
    bc = bytecode.ByteCode(func=func)
    bc.dump()

    interp = interpreter.Interpreter(bytecode=bc)
    interp.interpret()
    interp.dump(utils.StringIO())

    return interp


class TestLoopDetection(unittest.TestCase):

    def assertLoops(self, interp, expected):
        cfg = interp.cfa.graph
        self.assertEqual(len(cfg.loops()), expected)

    def test_sum1d(self):
        interp = interpret(usecases.sum1d)
        self.assertLoops(interp, 1)

    def test_sum2d(self):
        interp = interpret(usecases.sum2d)
        self.assertLoops(interp, 2)

    def test_while_count(self):
        interp = interpret(usecases.while_count)
        self.assertLoops(interp, 1)

    def test_copy_arrays(self):
        interp = interpret(usecases.copy_arrays)
        self.assertLoops(interp, 1)

    def test_andor(self):
        interp = interpret(usecases.andor)
        self.assertLoops(interp, 0)


if __name__ == '__main__':
    unittest.main()
