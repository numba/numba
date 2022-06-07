import numpy as np
from numba.cuda.testing import (unittest, CUDATestCase, skip_unless_cc_53,
                                skip_on_cudasim)
from numba import cuda
from numba.core.types import f2
from numba.cuda import compile_ptx
import operator
import itertools


def simple_fp16add(ary, a, b):
    ary[0] = a + b


def simple_fp16_iadd(ary, a):
    ary[0] += a


def simple_fp16_isub(ary, a):
    ary[0] -= a


def simple_fp16_imul(ary, a):
    ary[0] *= a


def simple_fp16sub(ary, a, b):
    ary[0] = a - b


def simple_fp16mul(ary, a, b):
    ary[0] = a * b


def simple_fp16neg(ary, a):
    ary[0] = -a


def simple_fp16abs(ary, a):
    ary[0] = abs(a)


class TestOperatorModule(CUDATestCase):
    """
    Test if operator module is supported by the CUDA target.
    """
    def operator_template(self, op):
        @cuda.jit
        def foo(a, b):
            i = 0
            a[i] = op(a[i], b[i])

        a = np.ones(1)
        b = np.ones(1)
        res = a.copy()
        foo[1, 1](res, b)

        np.testing.assert_equal(res, op(a, b))

    def test_add(self):
        self.operator_template(operator.add)

    def test_sub(self):
        self.operator_template(operator.sub)

    def test_mul(self):
        self.operator_template(operator.mul)

    def test_truediv(self):
        self.operator_template(operator.truediv)

    def test_floordiv(self):
        self.operator_template(operator.floordiv)

    @skip_unless_cc_53
    def test_fp16_binary(self):
        functions = (simple_fp16add, simple_fp16sub, simple_fp16mul)
        ops = (operator.add, operator.sub, operator.mul)

        for fn, op in zip(functions, ops):
            with self.subTest(op=op):
                kernel = cuda.jit("void(f2[:], f2, f2)")(fn)

                expected = np.zeros(1, dtype=np.float16)
                got = np.zeros(1, dtype=np.float16)
                arg1 = np.random.random(1).astype(np.float16)
                arg2 = np.random.random(1).astype(np.float16)

                kernel[1, 1](got, arg1[0], arg2[0])
                expected = op(arg1, arg2)
                np.testing.assert_allclose(got[0], expected)

    @skip_on_cudasim('Compilation unsupported in the simulator')
    def test_fp16_binary_ptx(self):
        functions = (simple_fp16add, simple_fp16sub, simple_fp16mul)
        ops = (operator.add, operator.sub, operator.mul)
        opstring = ('add.f16', 'sub.f16', 'mul.f16')
        args = (f2[:], f2, f2)

        for fn, op, s in zip(functions, ops, opstring):
            with self.subTest(op=op):
                ptx, _ = compile_ptx(fn, args, cc=(5, 3))
                self.assertIn(s, ptx)

    @skip_unless_cc_53
    def test_mixed_fp16_binary_arithmetic(self):
        functions = (simple_fp16add, simple_fp16sub, simple_fp16mul)
        ops = (operator.add, operator.sub, operator.mul)
        types = (np.int8, np.int16, np.int32, np.int64,
                 np.float32, np.float64)
        for (fn, op), ty in itertools.product(zip(functions, ops), types):
            with self.subTest(op=op):
                kernel = cuda.jit(fn)

                
                arg1 = np.random.random(1).astype(np.float16)
                arg2 = (np.random.random(1) * 100).astype(ty)
                arg2_ty = np.result_type(np.float16, ty)

                expected = np.zeros(1, dtype=arg2_ty)
                got = np.zeros(1, dtype=arg2_ty)

                kernel[1, 1](got, arg1[0], arg2[0])
                expected = op(arg1, arg2)
                np.testing.assert_allclose(got[0], expected)
    
    @skip_on_cudasim('Compilation unsupported in the simulator')
    def test_fp16_inplace_binary_ptx(self):
        functions = (simple_fp16_iadd, simple_fp16_isub, simple_fp16_imul)
        ops = (operator.iadd, operator.isub, operator.imul)
        opstring = ('add.f16', 'sub.f16', 'mul.f16')
        args = (f2[:], f2)

        for fn, op, s in zip(functions, ops, opstring):
            with self.subTest(op=op):
                ptx, _ = compile_ptx(fn, args, cc=(5, 3))
                self.assertIn(s, ptx)

    @skip_unless_cc_53
    def test_fp16_inplace_binary(self):
        functions = (simple_fp16_iadd, simple_fp16_isub, simple_fp16_imul)
        ops = (operator.iadd, operator.isub, operator.imul)

        for fn, op in zip(functions, ops):
            with self.subTest(op=op):
                kernel = cuda.jit("void(f2[:], f2)")(fn)

                expected = np.zeros(1, dtype=np.float16)
                got_in_out = np.random.random(1).astype(np.float16)
                arg1 = np.random.random(1).astype(np.float16)

                kernel[1, 1](got_in_out, arg1[0])
                expected = op(got_in_out, arg1)
                np.testing.assert_allclose(got_in_out[0], expected)

    @skip_unless_cc_53
    def test_fp16_unary(self):
        functions = (simple_fp16neg, simple_fp16abs)
        ops = (operator.neg, operator.abs)

        for fn, op in zip(functions, ops):
            with self.subTest(op=op):
                kernel = cuda.jit("void(f2[:], f2)")(fn)

                expected = np.zeros(1, dtype=np.float16)
                got = np.zeros(1, dtype=np.float16)
                arg1 = np.random.random(1).astype(np.float16)

                kernel[1, 1](got, arg1[0])
                expected = op(arg1)
                np.testing.assert_allclose(got[0], expected)

    @skip_on_cudasim('Compilation unsupported in the simulator')
    def test_fp16_neg_ptx(self):
        args = (f2[:], f2)
        ptx, _ = compile_ptx(simple_fp16neg, args, cc=(5, 3))
        self.assertIn('neg.f16', ptx)

    @skip_on_cudasim('Compilation unsupported in the simulator')
    def test_fp16_abs_ptx(self):
        args = (f2[:], f2)
        ptx, _ = compile_ptx(simple_fp16abs, args, cc=(5, 3))
        if cuda.runtime.get_version() < (10, 2):
            self.assertRegex(ptx, r'and\.b16.*0x7FFF;')
        else:
            self.assertIn('abs.f16', ptx)


if __name__ == '__main__':
    unittest.main()
