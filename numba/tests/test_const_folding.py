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
    N += 2  # invalidate N
    return N + 3

def cf_9():
    i = 1
    j = 2
    k = 3  # the only constant
    for i, n in range(10):
        j += 0
    return i + k

def cf_10():
    i = j = 123
    return i + j

def cf_11(M):
    return M + 2

def cf_12(a):
    M = a
    return M + 2


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

    def test_cf_9(self):
        astree = self.run_pipeline(cf_9)
        print utils.pformat_ast(astree)
        names = list(self.iter_all(astree, ast.Name))
        nums = list(self.iter_all(astree, ast.Num))
        self.assertEqual(len(names), 8)
        self.assertEqual(len(nums), 6)

    def test_cf_10(self):
        astree = self.run_pipeline(cf_10)
        print utils.pformat_ast(astree)
        names = list(self.iter_all(astree, ast.Name))
        nums = list(self.iter_all(astree, ast.Num))
        self.assertEqual(len(names), 2)
        self.assertEqual(len(nums), 2)

    def test_cf_11(self):
        astree = self.run_pipeline(cf_11)
        print utils.pformat_ast(astree)
        names = list(self.iter_all(astree, ast.Name))
        nums = list(self.iter_all(astree, ast.Num))
        self.assertEqual(len(names), 2)
        self.assertEqual(len(nums), 1)
        self.assertEqual(nums[0].n, (2))
    
    def test_cf_12(self):
        astree = self.run_pipeline(cf_12)
        print utils.pformat_ast(astree)
        names = list(self.iter_all(astree, ast.Name))
        nums = list(self.iter_all(astree, ast.Num))
        self.assertEqual(len(names), 4)
        self.assertEqual(len(nums), 1)
        self.assertEqual(nums[0].n, (2))



if __name__ == '__main__':
    unittest.main()
