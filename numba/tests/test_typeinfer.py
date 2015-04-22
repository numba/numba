from __future__ import print_function, division, absolute_import
import os, sys, subprocess
import itertools
import numpy as np
from numba import unittest_support as unittest
from numba.compiler import compile_isolated
from numba import types, typeinfer, typing, jit


class TestArgRetCasting(unittest.TestCase):
    def test_arg_ret_casting(self):
        def foo(x):
            return x

        args = (types.int32,)
        return_type = types.float32
        cres = compile_isolated(foo, args, return_type)
        self.assertTrue(isinstance(cres.entry_point(123), float))
        self.assertEqual(cres.signature.args, args)
        self.assertEqual(cres.signature.return_type, return_type)

    def test_arg_ret_mismatch(self):
        def foo(x):
            return x

        args = (types.Array(types.int32, 1, 'C'),)
        return_type = types.float32
        try:
            cres = compile_isolated(foo, args, return_type)
        except typeinfer.TypingError as e:
            pass
        else:
            self.fail("Should complain about array casting to float32")

    def test_invalid_arg_type_forcing(self):
        def foo(iters):
            a = range(iters)
            return iters

        args = (types.uint32,)
        return_type = types.uint8
        cres = compile_isolated(foo, args, return_type)
        typemap = cres.type_annotation.typemap
        # Argument "iters" must be uint32
        self.assertEqual(typemap['iters'], types.uint32)


class TestTupleUnify(unittest.TestCase):
    def test_int_tuple_unify(self):
        """
        Test issue #493
        """

        def foo(an_int32, an_int64):
            a = an_int32, an_int32
            while True:  # infinite loop
                a = an_int32, an_int64
            return a

        args = (types.int32, types.int64)
        # Check if compilation is successful
        cres = compile_isolated(foo, args)


class TestUnify(unittest.TestCase):
    @staticmethod
    def _actually_test_complex_unify():
        def pyfunc(a):
            res = 0.0
            for i in range(len(a)):
                res += a[i]
            return res

        argtys = [types.Array(types.complex128, 1, 'C')]
        cres = compile_isolated(pyfunc, argtys)
        return (pyfunc, cres)

    def test_complex_unify_issue599(self):
        pyfunc, cres = self._actually_test_complex_unify()
        arg = np.array([1.0j])
        cfunc = cres.entry_point
        self.assertEqual(cfunc(arg), pyfunc(arg))

    def test_complex_unify_issue599_multihash(self):
        """
        Test issue #599 for multiple values of PYTHONHASHSEED.
        """
        env = os.environ.copy()
        for seedval in (1, 2, 1024):
            env['PYTHONHASHSEED'] = str(seedval)
            subproc = subprocess.Popen(
                [sys.executable, '-c',
                 'import numba.tests.test_typeinfer as test_mod\n' +
                 'test_mod.TestUnify._actually_test_complex_unify()'],
                env=env)
            subproc.wait()
            self.assertEqual(subproc.returncode, 0, 'Child process failed.')

    def unify_pair_test(self, n):
        """
        Test all permutations of N-combinations of numeric types and ensure
        that the unification matches
        """
        ctx = typing.Context()
        for tys in itertools.combinations(types.number_domain, n):
            res = [ctx.unify_types(*comb)
                   for comb in itertools.permutations(tys)]
            # All result must be equal
            first_result = res[0]
            for other in res[1:]:
                self.assertEqual(first_result, other)

    def test_unify_pair(self):
        self.unify_pair_test(2)
        self.unify_pair_test(3)

    def test_bitwidth_number_types(self):
        """All numeric types have bitwidth attribute
        """
        for ty in types.number_domain:
            self.assertTrue(hasattr(ty, "bitwidth"))

    def test_unify_to_optional(self):
        """Test unification to optional type
        """
        ctx = typing.Context()
        for tys in itertools.combinations(types.number_domain, 2):
            tys = list(tys) + [types.none]
            res = [ctx.unify_types(*comb)
                   for comb in itertools.permutations(tys)]
            # All result must be equal
            first_result = res[0]
            self.assertIsInstance(first_result, types.Optional)
            for other in res[1:]:
                self.assertEqual(first_result, other)


def issue_797(x0, y0, x1, y1, grid):
    nrows, ncols = grid.shape

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)

    sx = 0
    if x0 < x1:
        sx = 1
    else:
        sx = -1
    sy = 0
    if y0 < y1:
        sy = 1
    else:
        sy = -1

    err = dx - dy

    while True:
        if x0 == x1 and y0 == y1:
            break

        if 0 <= x0 < nrows and 0 <= y0 < ncols:
            grid[x0, y0] += 1

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def issue_1080(a, b):
    if not a:
        return True
    return b


class TestIssue(unittest.TestCase):
    def test_issue_797(self):
        """https://github.com/numba/numba/issues/797#issuecomment-58592401

        Undeterministic triggering of tuple coercion error
        """
        foo = jit(nopython=True)(issue_797)
        g = np.zeros(shape=(10, 10), dtype=np.int32)
        foo(np.int32(0), np.int32(0), np.int32(1), np.int32(1), g)

    def test_issue_1080(self):
        """https://github.com/numba/numba/issues/1080

        Erroneous promotion of boolean args to int64
        """
        foo = jit(nopython=True)(issue_1080)
        foo(True, False)


class TestCoercion(unittest.TestCase):
    """
    Test coercion of binary operations.
    """
    references = {
        ('uint8', 'uint8'): 'uint8',
        ('int8', 'int8'): 'int8',
        ('uint16', 'uint16'): 'uint16',
        ('int16', 'int16'): 'int16',
        ('uint32', 'uint32'): 'uint32',
        ('int32', 'int32'): 'int32',
        ('uint64', 'uint64'): 'uint64',
        ('int64', 'int64'): 'int64',

        ('int8', 'uint8'): 'int16',
        ('int8', 'uint16'): 'int32',
        ('int8', 'uint32'): 'int64',

        ('uint8', 'int32'): 'int32',
        ('uint8', 'uint64'): 'uint64',

        ('int16', 'int8'): 'int16',
        ('int16', 'uint8'): 'int16',
        ('int16', 'uint16'): 'int32',
        ('int16', 'uint32'): 'int64',
        ('int16', 'int64'): 'int64',
        ('int16', 'uint64'): 'float64',

        ('uint16', 'uint8'): 'uint16',
        ('uint16', 'uint32'): 'uint32',
        ('uint16', 'int32'): 'int32',
        ('uint16', 'uint64'): 'uint64',

        ('int32', 'int8'): 'int32',
        ('int32', 'int16'): 'int32',
        ('int32', 'uint32'): 'int64',
        ('int32', 'int64'): 'int64',

        ('uint32', 'uint8'): 'uint32',
        ('uint32', 'int64'): 'int64',
        ('uint32', 'uint64'): 'uint64',

        ('int64', 'int8'): 'int64',
        ('int64', 'uint8'): 'int64',
        ('int64', 'uint16'): 'int64',

        ('uint64', 'int8'): 'float64',
        ('uint64', 'int32'): 'float64',
        ('uint64', 'int64'): 'float64',
    }

    def test_integer(self):
        ctx = typing.Context()
        for ut, st in itertools.product(types.integer_domain,
                                        types.integer_domain):
            unified = ctx.unify_types(ut, st)
            self._check_unify(ut, st, unified)

    def _check_unify(self, aty, bty, unified):
        key = (str(aty), str(bty))
        expect = self.references.get(key,
                                     self.references.get(tuple(reversed(key))))
        msg = "{0}, {1} -> {2} != {3}".format(aty, bty, unified, expect)
        self.assertEqual(str(unified), expect, msg=msg)


if __name__ == '__main__':
    unittest.main()
