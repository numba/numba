from __future__ import division

import itertools

import numpy as np
import sys
from collections import namedtuple

from numba import unittest_support as unittest
from numba import (njit, typeof, types, typing, typeof, ir, utils, bytecode,
    jitclass, prange)
from .support import TestCase, tag
from numba.array_analysis import EquivSet, ArrayAnalysis
from numba.compiler import Pipeline, Flags, _PipelineManager
from numba.targets import cpu, registry
from numba.numpy_support import version as numpy_version
from numba.ir_utils import remove_dead

# for parallel tests, marking that Windows with Python 2.7 is not supported
_windows_py27 = (sys.platform.startswith('win32') and
                 sys.version_info[:2] == (2, 7))
_32bit = sys.maxsize <= 2 ** 32
_reason = 'parfors not supported'
skip_unsupported = unittest.skipIf(_32bit or _windows_py27, _reason)


# test class for #3700
@jitclass([('L', types.int32), ('T', types.int32)])
class ExampleClass3700(object):
    def __init__(self, n):
        self.L = n
        self.T = n + 1


class TestEquivSet(TestCase):

    """
    Test array_analysis.EquivSet.
    """
    @tag('important')
    def test_insert_equiv(self):
        s1 = EquivSet()
        s1.insert_equiv('a', 'b')
        self.assertTrue(s1.is_equiv('a', 'b'))
        self.assertTrue(s1.is_equiv('b', 'a'))
        s1.insert_equiv('c', 'd')
        self.assertTrue(s1.is_equiv('c', 'd'))
        self.assertFalse(s1.is_equiv('c', 'a'))
        s1.insert_equiv('a', 'c')
        self.assertTrue(s1.is_equiv('a', 'b', 'c', 'd'))
        self.assertFalse(s1.is_equiv('a', 'e'))

    @tag('important')
    def test_intersect(self):
        s1 = EquivSet()
        s2 = EquivSet()
        r = s1.intersect(s2)
        self.assertTrue(r.is_empty())
        s1.insert_equiv('a', 'b')
        r = s1.intersect(s2)
        self.assertTrue(r.is_empty())
        s2.insert_equiv('b', 'c')
        r = s1.intersect(s2)
        self.assertTrue(r.is_empty())
        s2.insert_equiv('d', 'a')
        r = s1.intersect(s2)
        self.assertTrue(r.is_empty())
        s1.insert_equiv('a', 'e')
        s2.insert_equiv('c', 'd')
        r = s1.intersect(s2)
        self.assertTrue(r.is_equiv('a', 'b'))
        self.assertFalse(r.is_equiv('a', 'e'))
        self.assertFalse(r.is_equiv('c', 'd'))


class ArrayAnalysisTester(Pipeline):

    @classmethod
    def mk_pipeline(cls, args, return_type=None, flags=None, locals={},
                    library=None, typing_context=None, target_context=None):
        if not flags:
            flags = Flags()
        flags.nrt = True
        if typing_context is None:
            typing_context = registry.cpu_target.typing_context
        if target_context is None:
            target_context =  registry.cpu_target.target_context
        return cls(typing_context, target_context, library, args, return_type,
                   flags, locals)

    def compile_to_ir(self, func, test_idempotence=None):
        """
        Populate and run compiler pipeline
        """
        self.func_id = bytecode.FunctionIdentity.from_function(func)

        try:
            bc = self.extract_bytecode(self.func_id)
        except BaseException as e:
            raise e

        self.bc = bc
        self.lifted = ()
        self.lifted_from = None

        pm = _PipelineManager()

        pm.create_pipeline("nopython")
        if self.func_ir is None:
            pm.add_stage(self.stage_analyze_bytecode, "analyzing bytecode")
        pm.add_stage(self.stage_process_ir, "processing IR")
        if not self.flags.no_rewrites:
            if self.status.can_fallback:
                pm.add_stage(
                    self.stage_preserve_ir, "preserve IR for fallback")
            pm.add_stage(self.stage_generic_rewrites, "nopython rewrites")
        pm.add_stage(
            self.stage_inline_pass, "inline calls to locally defined closures")
        pm.add_stage(self.stage_nopython_frontend, "nopython frontend")
        pm.add_stage(self.stage_annotate_type, "annotate type")
        if not self.flags.no_rewrites:
            pm.add_stage(self.stage_nopython_rewrites, "nopython rewrites")
        func_ir_copies = []

        def stage_array_analysis():
            self.array_analysis = ArrayAnalysis(self.typingctx, self.func_ir,
                                                self.type_annotation.typemap,
                                                self.type_annotation.calltypes)
            self.array_analysis.run(self.func_ir.blocks)
            func_ir_copies.append(self.func_ir.copy())
            if test_idempotence and len(func_ir_copies) > 1:
                test_idempotence(func_ir_copies)

        pm.add_stage(stage_array_analysis, "analyze array equivalences")
        if test_idempotence:
            # Do another pass of array analysis to test idempotence
            pm.add_stage(stage_array_analysis, "analyze array equivalences")

        pm.finalize()
        res = pm.run(self.status)
        return self.array_analysis


class TestArrayAnalysis(TestCase):

    def compare_ir(self, ir_list):
        outputs = []
        for func_ir in ir_list:
            remove_dead(func_ir.blocks, func_ir.arg_names, func_ir)
            output = utils.StringIO()
            func_ir.dump(file=output)
            outputs.append(output.getvalue())
        self.assertTrue(len(set(outputs)) == 1)  # assert all outputs are equal

    def _compile_and_test(self, fn, arg_tys, asserts=[], equivs=[], idempotent=True):
        """
        Compile the given function and get its IR.
        """
        test_pipeline = ArrayAnalysisTester.mk_pipeline(arg_tys)
        test_idempotence = self.compare_ir if idempotent else lambda x:()
        analysis = test_pipeline.compile_to_ir(fn, test_idempotence)
        if equivs:
            for func in equivs:
                # only test the equiv_set of the first block
                func(analysis.equiv_sets[0])
        if asserts == None:
            self.assertTrue(self._has_no_assertcall(analysis.func_ir))
        else:
            for func in asserts:
                func(analysis.func_ir, analysis.typemap)

    def _has_assertcall(self, func_ir, typemap, args):
        msg = "Sizes of {} do not match".format(', '.join(args))
        for label, block in func_ir.blocks.items():
            for expr in block.find_exprs(op='call'):
                fn = func_ir.get_definition(expr.func.name)
                if isinstance(fn, ir.Global) and fn.name == 'assert_equiv':
                    typ = typemap[expr.args[0].name]
                    if typ.literal_value.startswith(msg):
                        return True
        return False

    def _has_shapecall(self, func_ir, x):
        for label, block in func_ir.blocks.items():
            for expr in block.find_exprs(op='getattr'):
                if expr.attr == 'shape':
                    y = func_ir.get_definition(expr.value, lhs_only=True)
                    z = func_ir.get_definition(x, lhs_only=True)
                    y = y.name if isinstance(y, ir.Var) else y
                    z = z.name if isinstance(z, ir.Var) else z
                    if y == z:
                        return True
        return False

    def _has_no_assertcall(self, func_ir):
        for label, block in func_ir.blocks.items():
            for expr in block.find_exprs(op='call'):
                fn = func_ir.get_definition(expr.func.name)
                if isinstance(fn, ir.Global) and fn.name == 'assert_equiv':
                    return False
        return True

    def with_assert(self, *args):
        return lambda func_ir, typemap: self.assertTrue(
            self._has_assertcall(func_ir, typemap, args))

    def without_assert(self, *args):
        return lambda func_ir, typemap: self.assertFalse(
            self._has_assertcall(func_ir, typemap, args))

    def with_equiv(self, *args):
        def check(equiv_set):
            n = len(args)
            for i in range(n - 1):
                if not equiv_set.is_equiv(args[i], args[n - 1]):
                    return False
            return True
        return lambda equiv_set: self.assertTrue(check(equiv_set))

    def without_equiv(self, *args):
        def check(equiv_set):
            n = len(args)
            for i in range(n - 1):
                if equiv_set.is_equiv(args[i], args[n - 1]):
                    return False
            return True
        return lambda equiv_set: self.assertTrue(check(equiv_set))

    def with_shapecall(self, x):
        return lambda func_ir, s: self.assertTrue(self._has_shapecall(func_ir, x))

    def without_shapecall(self, x):
        return lambda func_ir, s: self.assertFalse(self._has_shapecall(func_ir, x))

    def test_base_cases(self):
        def test_0():
            a = np.zeros(0)
            b = np.zeros(1)
            m = 0
            n = 1
            c = np.zeros((m, n))
            return
        self._compile_and_test(test_0, (),
                               equivs=[self.with_equiv('a', (0,)),
                                       self.with_equiv('b', (1,)),
                                       self.with_equiv('c', (0, 1))])

        def test_1(n):
            a = np.zeros(n)
            b = np.zeros(n)
            return a + b
        self._compile_and_test(test_1, (types.intp,), asserts=None)

        def test_2(m, n):
            a = np.zeros(n)
            b = np.zeros(m)
            return a + b
        self._compile_and_test(test_2, (types.intp, types.intp),
                               asserts=[self.with_assert('a', 'b')])

        def test_3(n):
            a = np.zeros(n)
            return a + n
        self._compile_and_test(test_3, (types.intp,), asserts=None)

        def test_4(n):
            a = np.zeros(n)
            b = a + 1
            c = a + 2
            return a + c
        self._compile_and_test(test_4, (types.intp,), asserts=None)

        def test_5(n):
            a = np.zeros((n, n))
            m = n
            b = np.zeros((m, n))
            return a + b
        self._compile_and_test(test_5, (types.intp,), asserts=None)

        def test_6(m, n):
            a = np.zeros(n)
            b = np.zeros(m)
            d = a + b
            e = a - b
            return d + e
        self._compile_and_test(test_6, (types.intp, types.intp),
                               asserts=[self.with_assert('a', 'b'),
                                        self.without_assert('d', 'e')])

        def test_7(m, n):
            a = np.zeros(n)
            b = np.zeros(m)
            if m == 10:
                d = a + b
            else:
                d = a - b
            return d + a
        self._compile_and_test(test_7, (types.intp, types.intp),
                               asserts=[self.with_assert('a', 'b'),
                                        self.without_assert('d', 'a')])

        def test_8(m, n):
            a = np.zeros(n)
            b = np.zeros(m)
            if m == 10:
                d = b + a
            else:
                d = a + a
            return b + d
        self._compile_and_test(test_8, (types.intp, types.intp),
                               asserts=[self.with_assert('b', 'a'),
                                        self.with_assert('b', 'd')])

        def test_9(m):
            A = np.ones(m)
            s = 0
            while m < 2:
                m += 1
                B = np.ones(m)
                s += np.sum(A + B)
            return s
        self._compile_and_test(test_9, (types.intp,),
                               asserts=[self.with_assert('A', 'B')])

        def test_10(m, n):
            p = m - 1
            q = n + 1
            r = q + 1
            A = np.zeros(p)
            B = np.zeros(q)
            C = np.zeros(r)
            D = np.zeros(m)
            s = np.sum(A + B)
            t = np.sum(C + D)
            return s + t
        self._compile_and_test(test_10, (types.intp,types.intp,),
                               asserts=[self.with_assert('A', 'B'),
                                        self.without_assert('C', 'D')])

        def test_tup_arg(T):
            T2 = T
            return T2[0]

        int_arr_typ = types.Array(types.intp, 1, 'C')
        self._compile_and_test(test_tup_arg,
            (types.Tuple((int_arr_typ, int_arr_typ)),), asserts=None)

        T = namedtuple("T", ['a','b'])
        def test_namedtuple(n):
            r = T(n, n)
            return r[0]
        self._compile_and_test(test_namedtuple, (types.intp,),
                                equivs=[self.with_equiv('r', ('n', 'n'))],)

        # np.where is tricky since it returns tuple of arrays
        def test_np_where_tup_return(A):
            c = np.where(A)
            return len(c[0])

        self._compile_and_test(test_np_where_tup_return,
            (types.Array(types.intp, 1, 'C'),), asserts=None)

        def test_shape(A):
            (m, n) = A.shape
            B = np.ones((m, n))
            return A + B
        self._compile_and_test(test_shape, (types.Array(types.intp, 2, 'C'),),
                               asserts=None)

        def test_cond(l, m, n):
            A = np.ones(l)
            B = np.ones(m)
            C = np.ones(n)
            if l == m:
                r = np.sum(A + B)
            else:
                r = 0
            if m != n:
                s = 0
            else:
                s = np.sum(B + C)
            t = 0
            if l == m:
                if m == n:
                    t = np.sum(A + B + C)
            return r + s + t
        self._compile_and_test(test_cond, (types.intp, types.intp, types.intp),
                               asserts=None)

        def test_assert_1(m, n):
            assert(m == n)
            A = np.ones(m)
            B = np.ones(n)
            return np.sum(A + B)
        self._compile_and_test(test_assert_1, (types.intp, types.intp),
                               asserts=None)

        def test_assert_2(A, B):
            assert(A.shape == B.shape)
            return np.sum(A + B)

        self._compile_and_test(test_assert_2, (types.Array(types.intp, 1, 'C'),
                                               types.Array(types.intp, 1, 'C'),),
                               asserts=None)
        self._compile_and_test(test_assert_2, (types.Array(types.intp, 2, 'C'),
                                               types.Array(types.intp, 2, 'C'),),
                               asserts=None)
        # expected failure
        with self.assertRaises(AssertionError) as raises:
            self._compile_and_test(test_assert_2, (types.Array(types.intp, 1, 'C'),
                                                   types.Array(types.intp, 2, 'C'),),
                                   asserts=None)
        msg = "Dimension mismatch"
        self.assertIn(msg, str(raises.exception))


    def test_stencilcall(self):
        from numba import stencil
        @stencil
        def kernel_1(a):
            return 0.25 * (a[0,1] + a[1,0] + a[0,-1] + a[-1,0])

        def test_1(n):
            a = np.ones((n,n))
            b = kernel_1(a)
            return a + b

        self._compile_and_test(test_1, (types.intp,),
                               equivs=[self.with_equiv('a', 'b')],
                               asserts=[self.without_assert('a', 'b')])

        def test_2(n):
            a = np.ones((n,n))
            b = np.ones((n+1,n+1))
            kernel_1(a, out=b)
            return a

        self._compile_and_test(test_2, (types.intp,),
                               equivs=[self.without_equiv('a', 'b')])

        @stencil(standard_indexing=('c',))
        def kernel_2(a, b, c):
            return a[0,1,0] + b[0,-1,0] + c[0]

        def test_3(n):
            a = np.arange(64).reshape(4,8,2)
            b = np.arange(64).reshape(n,8,2)
            u = np.zeros(1)
            v = kernel_2(a, b, u)
            return v

        # standard indexed arrays are not considered in size equivalence
        self._compile_and_test(test_3, (types.intp,),
                               equivs=[self.with_equiv('a', 'b', 'v'),
                                       self.without_equiv('a', 'u')],
                               asserts=[self.with_assert('a', 'b')])

    def test_slice(self):
        def test_1(m, n):
            A = np.zeros(m)
            B = np.zeros(n)
            s = np.sum(A + B)
            C = A[1:m-1]
            D = B[1:n-1]
            t = np.sum(C + D)
            return s + t
        self._compile_and_test(test_1, (types.intp,types.intp,),
                               asserts=[self.with_assert('A', 'B'),
                                        self.without_assert('C', 'D')],
                               idempotent=False)

        def test_2(m):
            A = np.zeros(m)
            B = A[0:m-3]
            C = A[1:m-2]
            D = A[2:m-1]
            E = B + C
            return D + E
        self._compile_and_test(test_2, (types.intp,),
                               asserts=[self.without_assert('B', 'C'),
                                        self.without_assert('D', 'E')],
                               idempotent=False)

        def test_3(m):
            A = np.zeros((m,m))
            B = A[0:m-2,0:m-2]
            C = A[1:m-1,1:m-1]
            E = B + C
            return E
        self._compile_and_test(test_3, (types.intp,),
                               asserts=[self.without_assert('B', 'C')],
                               idempotent=False)

        def test_4(m):
            A = np.zeros((m,m))
            B = A[0:m-2,:]
            C = A[1:m-1,:]
            E = B + C
            return E
        self._compile_and_test(test_4, (types.intp,),
                               asserts=[self.without_assert('B', 'C')],
                               idempotent=False)

        def test_5(m,n):
            A = np.zeros(m)
            B = np.zeros(m)
            B[0:m-2] = A[1:m-1]
            C = np.zeros(n)
            D = A[1:m-1]
            C[0:n-2] = D
            # B and C are not necessarily of the same size because we can't
            # derive m == n from (m-2) % m == (n-2) % n
            return B + C
        self._compile_and_test(test_5, (types.intp,types.intp),
                               asserts=[self.without_assert('B', 'A'),
                                        self.with_assert('C', 'D'),
                                        self.with_assert('B', 'C')],
                               idempotent=False)

        def test_6(m):
            A = np.zeros((m,m))
            B = A[0:m-2,:-1]
            C = A[1:m-1,:-1]
            E = B + C
            return E
        self._compile_and_test(test_6, (types.intp,),
                               asserts=[self.without_assert('B', 'C')],
                               idempotent=False)

        def test_7(m):
            A = np.zeros((m,m))
            B = A[0:m-2,-3:-1]
            C = A[1:m-1,-4:-2]
            E = B + C
            return E
        self._compile_and_test(test_7, (types.intp,),
                               asserts=[self.without_assert('B', 'C')],
                               idempotent=False)

        def test_8(m):
            A = np.zeros((m,m))
            B = A[:m-2,0:]
            C = A[1:-1,:]
            E = B + C
            return E
        self._compile_and_test(test_8, (types.intp,),
                               asserts=[self.without_assert('B', 'C')],
                               idempotent=False)

        def test_9(m):
            # issues #3461 and #3554, checks equivalence on empty slices
            # and across binop
            A = np.zeros((m))
            B = A[:0] # B = array([], dtype=int64)
            C = A[1:]
            D = A[:-1:-1] # D = array([], dtype=int64)
            E = B + D
            F = E
            F += 1 # F = array([], dtype=int64)
            return A, C, F
        self._compile_and_test(test_9, (types.intp,),
                               equivs=[self.without_equiv('B', 'C'),
                                       self.with_equiv('A', 'm'),
                                       self.with_equiv('B', 'D'),
                                       self.with_equiv('F', 'D'),],)

    def test_numpy_calls(self):
        def test_zeros(n):
            a = np.zeros(n)
            b = np.zeros((n, n))
            c = np.zeros(shape=(n, n))
        self._compile_and_test(test_zeros, (types.intp,),
                               equivs=[self.with_equiv('a', 'n'),
                                       self.with_equiv('b', ('n', 'n')),
                                       self.with_equiv('b', 'c')])

        def test_0d_array(n):
            a = np.array(1)
            b = np.ones(2)
            return a + b
        self._compile_and_test(test_0d_array, (types.intp,),
                               equivs=[self.without_equiv('a', 'b')],
                               asserts=[self.without_shapecall('a')])

        def test_ones(n):
            a = np.ones(n)
            b = np.ones((n, n))
            c = np.ones(shape=(n, n))
        self._compile_and_test(test_ones, (types.intp,),
                               equivs=[self.with_equiv('a', 'n'),
                                       self.with_equiv('b', ('n', 'n')),
                                       self.with_equiv('b', 'c')])

        def test_empty(n):
            a = np.empty(n)
            b = np.empty((n, n))
            c = np.empty(shape=(n, n))
        self._compile_and_test(test_empty, (types.intp,),
                               equivs=[self.with_equiv('a', 'n'),
                                       self.with_equiv('b', ('n', 'n')),
                                       self.with_equiv('b', 'c')])

        def test_eye(n):
            a = np.eye(n)
            b = np.eye(N=n)
            c = np.eye(N=n, M=n)
            d = np.eye(N=n, M=n + 1)
        self._compile_and_test(test_eye, (types.intp,),
                               equivs=[self.with_equiv('a', ('n', 'n')),
                                       self.with_equiv('b', ('n', 'n')),
                                       self.with_equiv('b', 'c'),
                                       self.without_equiv('b', 'd')])

        def test_identity(n):
            a = np.identity(n)
        self._compile_and_test(test_identity, (types.intp,),
                               equivs=[self.with_equiv('a', ('n', 'n'))])

        def test_diag(n):
            a = np.identity(n)
            b = np.diag(a)
            c = np.diag(b)
            d = np.diag(a, k=1)
        self._compile_and_test(test_diag, (types.intp,),
                               equivs=[self.with_equiv('b', ('n',)),
                                       self.with_equiv('c', ('n', 'n'))],
                               asserts=[self.with_shapecall('d'),
                                        self.without_shapecall('c')])

        def test_array_like(a):
            b = np.empty_like(a)
            c = np.zeros_like(a)
            d = np.ones_like(a)
            e = np.full_like(a, 1)
            f = np.asfortranarray(a)

        self._compile_and_test(test_array_like, (types.Array(types.intp, 2, 'C'),),
                               equivs=[
                                   self.with_equiv('a', 'b', 'd', 'e', 'f')],
                               asserts=[self.with_shapecall('a'),
                                        self.without_shapecall('b')])

        def test_reshape(n):
            a = np.ones(n * n)
            b = a.reshape((n, n))
            return a.sum() + b.sum()
        self._compile_and_test(test_reshape, (types.intp,),
                               equivs=[self.with_equiv('b', ('n', 'n'))],
                               asserts=[self.without_shapecall('b')])


        def test_transpose(m, n):
            a = np.ones((m, n))
            b = a.T
            c = a.transpose()
            # Numba njit cannot compile explicit transpose call!
            # c = np.transpose(b)
        self._compile_and_test(test_transpose, (types.intp, types.intp),
                               equivs=[self.with_equiv('a', ('m', 'n')),
                                       self.with_equiv('b', ('n', 'm')),
                                       self.with_equiv('c', ('n', 'm'))])


        def test_transpose_3d(m, n, k):
            a = np.ones((m, n, k))
            b = a.T
            c = a.transpose()
            d = a.transpose(2,0,1)
            dt = a.transpose((2,0,1))
            e = a.transpose(0,2,1)
            et = a.transpose((0,2,1))
            # Numba njit cannot compile explicit transpose call!
            # c = np.transpose(b)
        self._compile_and_test(test_transpose_3d, (types.intp, types.intp, types.intp),
                               equivs=[self.with_equiv('a', ('m', 'n', 'k')),
                                       self.with_equiv('b', ('k', 'n', 'm')),
                                       self.with_equiv('c', ('k', 'n', 'm')),
                                       self.with_equiv('d', ('k', 'm', 'n')),
                                       self.with_equiv('dt', ('k', 'm', 'n')),
                                       self.with_equiv('e', ('m', 'k', 'n')),
                                       self.with_equiv('et', ('m', 'k', 'n'))])

        def test_random(n):
            a0 = np.random.rand(n)
            a1 = np.random.rand(n, n)
            b0 = np.random.randn(n)
            b1 = np.random.randn(n, n)
            c0 = np.random.ranf(n)
            c1 = np.random.ranf((n, n))
            c2 = np.random.ranf(size=(n, n))
            d0 = np.random.random_sample(n)
            d1 = np.random.random_sample((n, n))
            d2 = np.random.random_sample(size=(n, n))
            e0 = np.random.sample(n)
            e1 = np.random.sample((n, n))
            e2 = np.random.sample(size=(n, n))
            f0 = np.random.random(n)
            f1 = np.random.random((n, n))
            f2 = np.random.random(size=(n, n))
            g0 = np.random.standard_normal(n)
            g1 = np.random.standard_normal((n, n))
            g2 = np.random.standard_normal(size=(n, n))
            h0 = np.random.chisquare(10, n)
            h1 = np.random.chisquare(10, (n, n))
            h2 = np.random.chisquare(10, size=(n, n))
            i0 = np.random.weibull(10, n)
            i1 = np.random.weibull(10, (n, n))
            i2 = np.random.weibull(10, size=(n, n))
            j0 = np.random.power(10, n)
            j1 = np.random.power(10, (n, n))
            j2 = np.random.power(10, size=(n, n))
            k0 = np.random.geometric(0.1, n)
            k1 = np.random.geometric(0.1, (n, n))
            k2 = np.random.geometric(0.1, size=(n, n))
            l0 = np.random.exponential(10, n)
            l1 = np.random.exponential(10, (n, n))
            l2 = np.random.exponential(10, size=(n, n))
            m0 = np.random.poisson(10, n)
            m1 = np.random.poisson(10, (n, n))
            m2 = np.random.poisson(10, size=(n, n))
            n0 = np.random.rayleigh(10, n)
            n1 = np.random.rayleigh(10, (n, n))
            n2 = np.random.rayleigh(10, size=(n, n))
            o0 = np.random.normal(0, 1, n)
            o1 = np.random.normal(0, 1, (n, n))
            o2 = np.random.normal(0, 1, size=(n, n))
            p0 = np.random.uniform(0, 1, n)
            p1 = np.random.uniform(0, 1, (n, n))
            p2 = np.random.uniform(0, 1, size=(n, n))
            q0 = np.random.beta(0.1, 1, n)
            q1 = np.random.beta(0.1, 1, (n, n))
            q2 = np.random.beta(0.1, 1, size=(n, n))
            r0 = np.random.binomial(0, 1, n)
            r1 = np.random.binomial(0, 1, (n, n))
            r2 = np.random.binomial(0, 1, size=(n, n))
            s0 = np.random.f(0.1, 1, n)
            s1 = np.random.f(0.1, 1, (n, n))
            s2 = np.random.f(0.1, 1, size=(n, n))
            t0 = np.random.gamma(0.1, 1, n)
            t1 = np.random.gamma(0.1, 1, (n, n))
            t2 = np.random.gamma(0.1, 1, size=(n, n))
            u0 = np.random.lognormal(0, 1, n)
            u1 = np.random.lognormal(0, 1, (n, n))
            u2 = np.random.lognormal(0, 1, size=(n, n))
            v0 = np.random.laplace(0, 1, n)
            v1 = np.random.laplace(0, 1, (n, n))
            v2 = np.random.laplace(0, 1, size=(n, n))
            w0 = np.random.randint(0, 10, n)
            w1 = np.random.randint(0, 10, (n, n))
            w2 = np.random.randint(0, 10, size=(n, n))
            x0 = np.random.triangular(-3, 0, 10, n)
            x1 = np.random.triangular(-3, 0, 10, (n, n))
            x2 = np.random.triangular(-3, 0, 10, size=(n, n))

        last = ord('x') + 1
        vars1d = [('n',)] + [chr(x) + '0' for x in range(ord('a'), last)]
        vars2d = [('n', 'n')] + [chr(x) + '1' for x in range(ord('a'), last)]
        vars2d += [chr(x) + '1' for x in range(ord('c'), last)]
        self._compile_and_test(test_random, (types.intp,),
                               equivs=[self.with_equiv(*vars1d),
                                       self.with_equiv(*vars2d)])

        def test_concatenate(m, n):
            a = np.ones(m)
            b = np.ones(n)
            c = np.concatenate((a, b))
            d = np.ones((2, n))
            e = np.ones((3, n))
            f = np.concatenate((d, e))
            # Numba njit cannot compile concatenate with single array!
            # g = np.ones((3,4,5))
            # h = np.concatenate(g)
            i = np.ones((m, 2))
            j = np.ones((m, 3))
            k = np.concatenate((i, j), axis=1)
            l = np.ones((m, n))
            o = np.ones((m, n))
            p = np.concatenate((l, o))
            # Numba njit cannot support list argument!
            # q = np.concatenate([d, e])
        self._compile_and_test(test_concatenate, (types.intp, types.intp),
                               equivs=[self.with_equiv('f', (5, 'n')),
                                       #self.with_equiv('h', (3 + 4 + 5, )),
                                       self.with_equiv('k', ('m', 5))],
                               asserts=[self.with_shapecall('c'),
                                        self.without_shapecall('f'),
                                        self.without_shapecall('k'),
                                        self.with_shapecall('p')])

        def test_vsd_stack():
            k = np.ones((2,))
            l = np.ones((2, 3))
            o = np.ones((2, 3, 4))
            p = np.vstack((k, k))
            q = np.vstack((l, l))
            r = np.hstack((k, k))
            s = np.hstack((l, l))
            t = np.dstack((k, k))
            u = np.dstack((l, l))
            v = np.dstack((o, o))

        self._compile_and_test(test_vsd_stack, (),
                               equivs=[self.with_equiv('p', (2, 2)),
                                       self.with_equiv('q', (4, 3)),
                                       self.with_equiv('r', (4,)),
                                       self.with_equiv('s', (2, 6)),
                                       self.with_equiv('t', (1, 2, 2)),
                                       self.with_equiv('u', (2, 3, 2)),
                                       self.with_equiv('v', (2, 3, 8)),
                                       ])

        if numpy_version >= (1, 10):
            def test_stack(m, n):
                a = np.ones(m)
                b = np.ones(n)
                c = np.stack((a, b))
                d = np.ones((m, n))
                e = np.ones((m, n))
                f = np.stack((d, e))
                g = np.stack((d, e), axis=0)
                h = np.stack((d, e), axis=1)
                i = np.stack((d, e), axis=2)
                j = np.stack((d, e), axis=-1)

            self._compile_and_test(test_stack, (types.intp, types.intp),
                                   equivs=[self.with_equiv('m', 'n'),
                                           self.with_equiv('c', (2, 'm')),
                                           self.with_equiv(
                                       'f', 'g', (2, 'm', 'n')),
                self.with_equiv(
                                       'h', ('m', 2, 'n')),
                self.with_equiv(
                                       'i', 'j', ('m', 'n', 2)),
            ])

        def test_linspace(m, n):
            a = np.linspace(m, n)
            b = np.linspace(m, n, 10)
            # Numba njit does not support num keyword to linspace call!
            # c = np.linspace(m,n,num=10)
        self._compile_and_test(test_linspace, (types.float64, types.float64),
                               equivs=[self.with_equiv('a', (50,)),
                                       self.with_equiv('b', (10,))])

        def test_dot(l, m, n):
            a = np.dot(np.ones(1), np.ones(1))
            b = np.dot(np.ones(2), np.ones((2, 3)))
            # Numba njit does not support higher dimensional inputs
            #c = np.dot(np.ones(2),np.ones((3,2,4)))
            #d = np.dot(np.ones(2),np.ones((3,5,2,4)))
            e = np.dot(np.ones((1, 2)), np.ones(2,))
            #f = np.dot(np.ones((1,2,3)),np.ones(3,))
            #g = np.dot(np.ones((1,2,3,4)),np.ones(4,))
            h = np.dot(np.ones((2, 3)), np.ones((3, 4)))
            i = np.dot(np.ones((m, n)), np.ones((n, m)))
            j = np.dot(np.ones((m, m)), np.ones((l, l)))

        self._compile_and_test(test_dot, (types.intp, types.intp, types.intp),
                               equivs=[self.without_equiv('a', (1,)),  # not array
                                       self.with_equiv('b', (3,)),
                                       self.with_equiv('e', (1,)),
                                       self.with_equiv('h', (2, 4)),
                                       self.with_equiv('i', ('m', 'm')),
                                       self.with_equiv('j', ('m', 'm')),
                                       ],
                               asserts=[self.with_assert('m', 'l')])

        def test_broadcast(m, n):
            a = np.ones((m, n))
            b = np.ones(n)
            c = a + b
            d = np.ones((1, n))
            e = a + c - d
        self._compile_and_test(test_broadcast, (types.intp, types.intp),
                               equivs=[self.with_equiv('a', 'c', 'e')],
                               asserts=None)

class TestArrayAnalysisParallelRequired(TestCase):
    """This is to just split out tests that need the parallel backend and
    therefore serialised execution.
    """

    _numba_parallel_test_ = False

    @skip_unsupported
    def test_misc(self):

        @njit
        def swap(x, y):
            return(y, x)

        def test_bug2537(m):
            a = np.ones(m)
            b = np.ones(m)
            for i in range(m):
                a[i], b[i] = swap(a[i], b[i])

        try:
            njit(test_bug2537, parallel=True)(10)
        except IndexError:
            self.fail("test_bug2537 raised IndexError!")

    @skip_unsupported
    def test_global_namedtuple(self):
        Row = namedtuple('Row', ['A'])
        row = Row(3)

        def test_impl():
            rr = row
            res = rr.A
            if res == 2:
                res = 3
            return res

        self.assertEqual(njit(test_impl, parallel=True)(), test_impl())

    @skip_unsupported
    def test_array_T_issue_3700(self):

        def test_impl(t_obj, X):
            for i in prange(t_obj.T):
                X[i] = i
            return X.sum()

        n = 5
        t_obj = ExampleClass3700(n)
        X1 = np.zeros(t_obj.T)
        X2 = np.zeros(t_obj.T)
        self.assertEqual(
            njit(test_impl, parallel=True)(t_obj, X1), test_impl(t_obj, X2))

if __name__ == '__main__':
    unittest.main()
