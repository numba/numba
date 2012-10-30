import unittest
import ast, inspect
import numpy as np
from numba import utils, decorators
from numba.minivect import minitypes
from numba import *

def cf_1():
    return 1 + 2

def cf_2():
    return 1 + 2 - 3 * 4 / 5 // 6 ** 7

def cf_3():
    return 0xbad & 0xbeef | 0xcafe

def cf_4():
    return True or False

def cf_5():
    return 1 != 2 and 3 < 4 and 5 > 8 / 9

M = 1
def cf_6():
    return M + 2

def cf_7():
    N = 1
    return N + 2

def cf_8():
    N = 1
    N += 2
    return N + 3

class TestConstFolding(unittest.TestCase):
    def run_pipeline(self, func):
        func_sig = minitypes.FunctionType(minitypes.void, [])
        source = inspect.getsource(func)
        astree = ast.parse(source)
        pipeline = decorators.context.numba_pipeline(decorators.context,
                                                     func, astree, func_sig)
        return pipeline.const_folding(astree)

    def iter_all(self, astree, target):
        for node in ast.walk(astree):
            if isinstance(node, target):
                yield node

    def test_cf_1(self):
        astree = self.run_pipeline(cf_1)
        print utils.pformat_ast(astree)
        nums = list(self.iter_all(astree, ast.Num))
        self.assertEqual(len(nums), 1)
        self.assertEqual(nums[0].n, (1 + 2))

    def test_cf_2(self):
        astree = self.run_pipeline(cf_2)
        print utils.pformat_ast(astree)
        nums = list(self.iter_all(astree, ast.Num))
        self.assertEqual(len(nums), 1)
        self.assertEqual(nums[0].n, (1 + 2 - 3 * 4 / 5 // 6 ** 7))

    def test_cf_3(self):
        astree = self.run_pipeline(cf_3)
        print utils.pformat_ast(astree)
        nums = list(self.iter_all(astree, ast.Num))
        self.assertEqual(len(nums), 1)
        self.assertEqual(nums[0].n, (0xbad & 0xbeef | 0xcafe))

    def test_cf_4(self):
        astree = self.run_pipeline(cf_4)
        print utils.pformat_ast(astree)
        names = list(self.iter_all(astree, ast.Name))
        self.assertEqual(len(names), 1)
        self.assertEqual(names[0].id, 'True')

    def test_cf_5(self):
        astree = self.run_pipeline(cf_5)
        print utils.pformat_ast(astree)
        names = list(self.iter_all(astree, ast.Name))
        self.assertEqual(len(names), 1)
        self.assertEqual(names[0].id, str(1 != 2 and 3 < 4 and 5 > 8 / 9))

    def test_cf_6(self):
        astree = self.run_pipeline(cf_6)
        print utils.pformat_ast(astree)
        names = list(self.iter_all(astree, ast.Name))
        nums = list(self.iter_all(astree, ast.Num))
        self.assertEqual(len(names), 0)
        self.assertEqual(len(nums), 1)
        self.assertEqual(nums[0].n, (1 + 2))

    def test_cf_7(self):
        astree = self.run_pipeline(cf_7)
        print utils.pformat_ast(astree)
        names = list(self.iter_all(astree, ast.Name))
        nums = list(self.iter_all(astree, ast.Num))
        # No removal of constant assignment
        self.assertEqual(len(names), 1)
        self.assertEqual(len(nums), 2)
        self.assertEqual(nums[1].n, (1 + 2))

    def test_cf_8(self):
        astree = self.run_pipeline(cf_8)
        print utils.pformat_ast(astree)
        names = list(self.iter_all(astree, ast.Name))
        nums = list(self.iter_all(astree, ast.Num))
        self.assertEqual(len(names), 3)
        self.assertEqual(len(nums), 3)
        for name in names:
            self.assertEqual(name.id, 'N')
        for i, num in enumerate(nums):
            self.assertEqual(num.n, i + 1)

if __name__ == '__main__':
    unittest.main()
