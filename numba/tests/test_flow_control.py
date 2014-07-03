from __future__ import print_function

import itertools

import numba.unittest_support as unittest
from numba.compiler import compile_isolated, Flags
from numba import types
from .support import TestCase

enable_pyobj_flags = Flags()
enable_pyobj_flags.set("enable_pyobject")

no_pyobj_flags = Flags()


def for_loop_usecase1(x, y):
    result = 0
    for i in range(x):
        result += i
    return result

def for_loop_usecase2(x, y):
    result = 0
    for i, j in enumerate(range(x, y, -1)):
        result += i * j
    return result

def for_loop_usecase3(x, y):
    result = 0
    for i in [x,y]:
        result += i
    return result

def for_loop_usecase4(x, y):
    result = 0
    for i in range(10):
        for j in range(10):
            result += 1
    return result

def for_loop_usecase5(x, y):
    result = 0
    for i in range(x):
        result += 1
        if result > y:
            break
    return result

def for_loop_usecase6(x, y):
    result = 0
    for i in range(x):
        if i > y:
            continue
        result += 1
    return result

def for_loop_usecase7(x, y):
    for i in range(x):
        x = 0
        for j in range(x):
            return 1
        else:
            pass
    return 0

def for_loop_usecase8(x, y):
    result = 0
    for i in range(x, y, y - x + 1):
        result += 1
    return result

def for_loop_usecase9(x, y):
    z = 0
    for i in range(x):
        x = 0
        for j in range(x):
            if j == x / 2:
                z += j
                break
        else:
            z += y

    return z


def while_loop_usecase1(x, y):
    result = 0
    i = 0
    while i < x:
        result += i
        i += 1
    return result

def while_loop_usecase2(x, y):
    result = 0
    while result != x:
        result += 1
    return result

def while_loop_usecase3(x, y):
    result = 0
    i = 0
    j = 0
    while i < x:
        while j < y:
            result += i + j
            i += 1
            j += 1
    return result

def while_loop_usecase4(x, y):
    result = 0
    while True:
        result += 1
        if result > x:
            break
    return result

def while_loop_usecase5(x, y):
    result = 0
    while result < x:
        if result > y:
            result += 2
            continue
        result += 1
    return result

def ifelse_usecase1(x, y):
    if x > 0:
        pass
    elif y > 0:
        pass
    else:
        pass
    return True

def ifelse_usecase2(x, y):
    if x > y:
        return 1
    elif x == 0 or y == 0:
        return 2
    else:
        return 3

def ifelse_usecase3(x, y):
    if x > 0:
        if y > 0:
            return 1
        elif y < 0:
            return 1
        else:
            return 0
    elif x < 0:
        return 1
    else:
        return 0

def ifelse_usecase4(x, y):
    if x == y:
        return 1

def ternary_ifelse_usecase1(x, y):
    return True if x > y else False


class TestFlowControl(TestCase):

    def run_test(self, pyfunc, x_operands, y_operands,
                 flags=enable_pyobj_flags):
        cr = compile_isolated(pyfunc, (types.int32, types.int32), flags=flags)
        cfunc = cr.entry_point
        for x, y in itertools.product(x_operands, y_operands):
            pyerr = None
            cerr = None
            try:
                pyres = pyfunc(x, y)
            except Exception as e:
                print("note: ", pyfunc, (x, y), "raises exception: %s" % e)
                pyerr = e

            try:
                cres = cfunc(x, y)
            except Exception as e:
                print("note: ", cfunc, (x, y), "raises exception: %s" % e)
                if pyerr is None:
                    raise
                cerr = e
            else:
                if pyerr is not None:
                    self.fail("Invalid for pure-python but numba works\n" +
                              pyerr)
            self.assertEqual(pyres, cres)

    def test_for_loop1(self, flags=enable_pyobj_flags):
        self.run_test(for_loop_usecase1, [-10, 0, 10], [0], flags=flags)

    def test_for_loop1_npm(self):
        self.test_for_loop1(flags=no_pyobj_flags)

    def test_for_loop2(self, flags=enable_pyobj_flags):
        self.run_test(for_loop_usecase2, [-10, 0, 10], [-10, 0, 10],
                      flags=flags)

    def test_for_loop2_npm(self):
        self.test_for_loop2(flags=no_pyobj_flags)

    def test_for_loop3(self, flags=enable_pyobj_flags):
        """
        List requires pyobject
        """
        self.run_test(for_loop_usecase3, [1], [2],
                      flags=flags)

    def test_for_loop3_npm(self):
        with self.assertTypingError():
            self.test_for_loop3(flags=no_pyobj_flags)

    def test_for_loop4(self, flags=enable_pyobj_flags):
        self.run_test(for_loop_usecase4, [10], [10], flags=flags)

    def test_for_loop4_npm(self):
        self.test_for_loop4(flags=no_pyobj_flags)

    def test_for_loop5(self, flags=enable_pyobj_flags):
        self.run_test(for_loop_usecase5, [100], [50], flags=flags)

    def test_for_loop5_npm(self):
        self.test_for_loop5(flags=no_pyobj_flags)

    def test_for_loop6(self, flags=enable_pyobj_flags):
        self.run_test(for_loop_usecase6, [100], [50], flags=flags)

    def test_for_loop6_npm(self):
        self.test_for_loop6(flags=no_pyobj_flags)

    def test_for_loop7(self, flags=enable_pyobj_flags):
        self.run_test(for_loop_usecase7, [5], [0], flags=flags)

    def test_for_loop7_npm(self):
        self.test_for_loop7(flags=no_pyobj_flags)

    def test_for_loop8(self, flags=enable_pyobj_flags):
        self.run_test(for_loop_usecase8, [0, 1], [0, 2, 10], flags=flags)

    def test_for_loop9(self, flags=enable_pyobj_flags):
        self.run_test(for_loop_usecase9, [0, 1], [0, 2, 10], flags=flags)

    def test_for_loop8_npm(self):
        self.test_for_loop8(flags=no_pyobj_flags)

    def test_while_loop1(self, flags=enable_pyobj_flags):
        self.run_test(while_loop_usecase1, [10], [0], flags=flags)

    def test_while_loop1_npm(self):
        self.test_while_loop1(flags=no_pyobj_flags)

    def test_while_loop2(self, flags=enable_pyobj_flags):
        self.run_test(while_loop_usecase2, [10], [0], flags=flags)

    def test_while_loop2_npm(self):
        self.test_while_loop2(flags=no_pyobj_flags)

    def test_while_loop3(self, flags=enable_pyobj_flags):
        self.run_test(while_loop_usecase3, [10], [10], flags=flags)

    def test_while_loop3_npm(self):
        self.test_while_loop3(flags=no_pyobj_flags)

    def test_while_loop4(self, flags=enable_pyobj_flags):
        self.run_test(while_loop_usecase4, [10], [0], flags=flags)

    def test_while_loop4_npm(self):
        self.test_while_loop4(flags=no_pyobj_flags)

    def test_while_loop5(self, flags=enable_pyobj_flags):
        self.run_test(while_loop_usecase5, [0, 5, 10], [0, 5, 10], flags=flags)

    def test_while_loop5_npm(self):
        self.test_while_loop5(flags=no_pyobj_flags)

    def test_ifelse1(self, flags=enable_pyobj_flags):
        self.run_test(ifelse_usecase1, [-1, 0, 1], [-1, 0, 1], flags=flags)

    def test_ifelse1_npm(self):
        self.test_ifelse1(flags=no_pyobj_flags)

    def test_ifelse2(self, flags=enable_pyobj_flags):
        self.run_test(ifelse_usecase2, [-1, 0, 1], [-1, 0, 1], flags=flags)

    def test_ifelse2_npm(self):
        self.test_ifelse2(flags=no_pyobj_flags)

    def test_ifelse3(self, flags=enable_pyobj_flags):
        self.run_test(ifelse_usecase3, [-1, 0, 1], [-1, 0, 1], flags=flags)

    def test_ifelse3_npm(self):
        self.test_ifelse3(flags=no_pyobj_flags)

    def test_ifelse4(self, flags=enable_pyobj_flags):
        self.run_test(ifelse_usecase4, [-1, 0, 1], [-1, 0, 1], flags=flags)

    def test_ifelse4_npm(self):
        self.test_ifelse4(flags=no_pyobj_flags)

    def test_ternary_ifelse1(self, flags=enable_pyobj_flags):
        self.run_test(ternary_ifelse_usecase1, [-1, 0, 1], [-1, 0, 1],
                      flags=flags)

    def test_ternary_ifelse1_npm(self):
        self.test_ternary_ifelse1(flags=no_pyobj_flags)

if __name__ == '__main__':
    unittest.main(verbosity=2)

