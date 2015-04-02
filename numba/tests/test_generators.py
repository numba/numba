from __future__ import print_function

import numpy as np

from numba.compiler import compile_isolated, Flags
from numba import jit, types
from numba.tests.support import TestCase
import numba.unittest_support as unittest
from numba import testing


enable_pyobj_flags = Flags()
enable_pyobj_flags.set("enable_pyobject")

forceobj_flags = Flags()
forceobj_flags.set("force_pyobject")

no_pyobj_flags = Flags()


def make_consumer(gen_func):
    def consumer(x):
        res = 0.0
        for y in gen_func(x):
            res += y
        return res
    return consumer


def gen1(x):
    for i in range(x):
        yield i

def gen2(x):
    for i in range(x):
        yield i
        for j in range(1, 3):
            yield i + j

def gen3(x):
    # Polymorphic yield types must be unified
    yield x
    yield x + 1.5
    yield x + 1j

def gen4(x, y, z):
    for i in range(3):
        yield z
        yield y + z
    return
    yield x

def gen5():
    # The bytecode for this generator doesn't contain any YIELD_VALUE
    # (it's optimized away).  We fail typing it, since the yield type
    # is entirely undefined.
    if 0:
        yield 1

def gen6(a, b):
    # Infinite loop: exercise computation of state variables
    x = a + 1
    while True:
        y = b + 2
        yield x + y

def gen7(arr):
    # Array variable in generator state
    for i in range(arr.size):
        yield arr[i]


def genobj(x):
    object()
    yield x


def return_generator_expr(x):
    return (i*2 for i in x)


class TestGenerators(TestCase):

    def check_generator(self, pygen, cgen):
        self.assertEqual(next(cgen), next(pygen))
        # Use list comprehensions to make sure we trash the generator's
        # former C stack.
        expected = [x for x in pygen]
        got = [x for x in cgen]
        self.assertEqual(expected, got)
        with self.assertRaises(StopIteration):
            next(cgen)

    def check_gen1(self, flags=no_pyobj_flags):
        pyfunc = gen1
        cr = compile_isolated(pyfunc, (types.int32,), flags=flags)
        pygen = pyfunc(8)
        cgen = cr.entry_point(8)
        self.check_generator(pygen, cgen)

    def test_gen1(self):
        self.check_gen1()

    def test_gen1_objmode(self):
        self.check_gen1(flags=forceobj_flags)

    def check_gen2(self, flags=no_pyobj_flags):
        pyfunc = gen2
        cr = compile_isolated(pyfunc, (types.int32,), flags=flags)
        pygen = pyfunc(8)
        cgen = cr.entry_point(8)
        self.check_generator(pygen, cgen)

    def test_gen2(self):
        self.check_gen2()

    def test_gen2_objmode(self):
        self.check_gen2(flags=forceobj_flags)

    def check_gen3(self, flags=no_pyobj_flags):
        pyfunc = gen3
        cr = compile_isolated(pyfunc, (types.int32,), flags=flags)
        pygen = pyfunc(8)
        cgen = cr.entry_point(8)
        self.check_generator(pygen, cgen)

    def test_gen3(self):
        self.check_gen3()

    def test_gen3_objmode(self):
        self.check_gen3(flags=forceobj_flags)

    def check_gen4(self, flags=no_pyobj_flags):
        pyfunc = gen4
        cr = compile_isolated(pyfunc, (types.int32,) * 3, flags=flags)
        pygen = pyfunc(5, 6, 7)
        cgen = cr.entry_point(5, 6, 7)
        self.check_generator(pygen, cgen)

    def test_gen4(self):
        self.check_gen4()

    def test_gen4_objmode(self):
        self.check_gen4(flags=forceobj_flags)

    def test_gen5(self):
        with self.assertTypingError() as cm:
            cr = compile_isolated(gen5, ())
        self.assertIn("Cannot type generator: it does not yield any value",
                      str(cm.exception))

    def test_gen5_objmode(self):
        cr = compile_isolated(gen5, (), flags=forceobj_flags)
        cgen = cr.entry_point()
        self.assertEqual(list(cgen), [])
        with self.assertRaises(StopIteration):
            next(cgen)

    def check_gen6(self, flags=no_pyobj_flags):
        pyfunc = gen6
        cr = compile_isolated(pyfunc, (types.int32,) * 2, flags=flags)
        cgen = cr.entry_point(5, 6)
        l = []
        for i in range(3):
            l.append(next(cgen))
        self.assertEqual(l, [14] * 3)

    def test_gen6(self):
        self.check_gen6()

    def test_gen6_objmode(self):
        self.check_gen6(flags=forceobj_flags)

    def check_gen7(self, flags=no_pyobj_flags):
        pyfunc = gen7
        cr = compile_isolated(pyfunc, (types.Array(types.float64, 1, 'C'),),
                              flags=flags)
        arr = np.linspace(1, 10, 7)
        pygen = pyfunc(arr.copy())
        cgen = cr.entry_point(arr)
        self.check_generator(pygen, cgen)

    def test_gen7(self):
        self.check_gen7()

    def test_gen7_objmode(self):
        self.check_gen7(flags=forceobj_flags)

    def check_consume_generator(self, gen_func):
        cgen = jit(nopython=True)(gen_func)
        cfunc = jit(nopython=True)(make_consumer(cgen))
        pyfunc = make_consumer(gen_func)
        expected = pyfunc(5)
        got = cfunc(5)
        self.assertPreciseEqual(got, expected)

    def test_consume_gen1(self):
        self.check_consume_generator(gen1)

    def test_consume_gen2(self):
        self.check_consume_generator(gen2)

    def test_consume_gen3(self):
        self.check_consume_generator(gen3)


class TestGenExprs(TestCase):

    @testing.allow_interpreter_mode
    def test_return_generator_expr(self):
        pyfunc = return_generator_expr
        cr = compile_isolated(pyfunc, ())
        cfunc = cr.entry_point
        self.assertEqual(sum(cfunc([1, 2, 3])), sum(pyfunc([1, 2, 3])))


if __name__ == '__main__':
    unittest.main()
