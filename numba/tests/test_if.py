#! /usr/bin/env python
# ______________________________________________________________________
'''test_if

Test phi node (or similar) generation for CFG joins beyond
if-then-else statements.
'''
# ______________________________________________________________________

from numba import *

import unittest

# ______________________________________________________________________

def if_fn_1(arg):
    if arg > 0.:
        result = 22.
    else:
        result = 42.
    return result


def if_fn_2(i, j):
    n = 5
    m = 5
    if j >= 1 and j < n - 1 and i >= 1 and i < m - 1:
        return i + j
    return 0xcafe

def if_fn_3(i, j):
    n = 5
    m = 5
    if j >= 1:
        if j < n - 1:
            if i >= 1:
                if i < m - 1:
                    return i + j
    return 0xbeef

def if_fn_4(i, j):
    if i < 0 or j < 0:
        return i + j
    return 0xdead

def if_fn_5(i, j):
    if i < 0:
        return i + j
    if j < 0:
        return i + j
    return 0xdefaced

def if_fn_6(i, j):
    if i < j:
        return  #
    i += j

def if_fn_7(i):
    if i:
        return i + 1
    return i

def if_fn_8(i, j):
    if i > j:
        return 1
    return 0

def if_fn_9(i, j, k):
    if i or (j and k):
        return 1
    return 0

@autojit
def if_bool(b):
    if b:
        return 1
    else:
        return 2

@autojit
def if_bool_constant_true():
    if True:
        return 1
    else:
        return 2

@autojit
def if_bool_constant_false():
    if False:
        return 1
    else:
        return 2


# ______________________________________________________________________

class TestIf(unittest.TestCase):
    def test_if_fn_1(self):
        if_fn_1c = jit(restype=f4, argtypes=[f4], backend='ast')(if_fn_1)
        oracle = if_fn_1
        self.assertEqual(if_fn_1c(-1.), if_fn_1(-1.))
        self.assertEqual(if_fn_1c(1.),  if_fn_1(1.))

    def test_if_fn_2(self):
        if_fn_2c = jit(restype=i4, argtypes=[i4, i4], backend='ast')(if_fn_2)

        oracle = if_fn_2
        for i in range(6):
            for j in range(6):
                self.assertEqual(if_fn_2c(i, j), oracle(i, j))

    def test_if_fn_3(self):
        if_fn_3c = jit(restype=i4, argtypes=[i4, i4], backend='ast')(if_fn_3)

        oracle = if_fn_3
        for i in range(6):
            for j in range(6):
                self.assertEqual(if_fn_3c(i, j), oracle(i, j))

    def test_if_fn_4(self):
        from meta.decompiler import decompile_func
        import ast, inspect
        if_fn_4c = jit(restype=i4, argtypes=[i4, i4], backend='ast')(if_fn_4)

        oracle = if_fn_4
        for i in range(-3, 3):
            for j in range(-3, 3):
                self.assertEqual(if_fn_4c(i, j), oracle(i, j))

    def test_if_fn_5(self):
        if_fn_5c = jit(restype=i4, argtypes=[i4, i4], backend='ast')(if_fn_5)
        oracle = if_fn_5
        for i in range(-3, 3):
            for j in range(-3, 3):
                self.assertEqual(if_fn_5c(i, j), oracle(i, j))

    def test_if_fn_6(self):
        if_fn_6c = jit(restype=void, argtypes=[i4, i4], backend='ast')(if_fn_6)

    def test_if_fn_7(self):
        # if_fn_7c = jit(restype=i4, argtypes=[i4], backend='ast')(if_fn_7)
        if_fn_7c = autojit(if_fn_7)
        oracle = if_fn_7
        for i in range(-3, 3):
            self.assertEqual(if_fn_7c(i), oracle(i))
            self.assertEqual(if_fn_7c(float(i)), oracle(float(i)))
            # self.assertEqual(if_fn_7c(i+1j), oracle(i+1j))

    def test_if_fn_8(self):
        if_fn_5c = jit(restype=i4, argtypes=[i4, i4], backend='ast')(if_fn_8)
        oracle = if_fn_8
        for i in range(-3, 3):
            for j in range(-3, 3):
                self.assertEqual(if_fn_5c(i, j), oracle(i, j))

    def test_if_fn_9(self):
        if_fn_5c = jit(restype=i4, argtypes=[i4, i4, i4], backend='ast')(
            if_fn_9)
        oracle = if_fn_9
        for i in range(-2, 2):
            for j in range(-2, 2):
                for k in range(-2, 2):
                    self.assertEqual(if_fn_5c(i, j, k), oracle(i, j, k))

    def test_if_bool(self):
        self.assertEqual(if_bool(True), 1)
        self.assertEqual(if_bool(False), 2)
        self.assertEqual(if_bool_constant_true(), 1)
        self.assertEqual(if_bool_constant_false(), 2)

# ______________________________________________________________________

if __name__ == "__main__":
#    if_fn_1c = jit(restype=f4, argtypes=[f4], backend='ast')(if_fn_1)
#    if_fn_4c = jit(restype=i4, argtypes=[i4, i4], backend='ast')(if_fn_4)
#    if_fn_5c = jit(restype=i4, argtypes=[i4, i4], backend='ast')(if_fn_5)
#    if_fn_6c = jit(restype=void, argtypes=[i4, i4], backend='ast')(if_fn_6)
#    if_fn_7c = jit(restype=i4, argtypes=[i4], backend='ast')(if_fn_7)
#    print if_fn_7c(-2), if_fn_7(-2)
#    print if_bool_constant_true()
    unittest.main()

# ______________________________________________________________________
# End of test_if.py
