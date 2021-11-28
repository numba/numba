import unittest
import pickle

import numpy as np
import numpy.core.umath_tests as ut

from numba import void, float32, int64, jit, guvectorize
from numba.np.ufunc import GUVectorize
from numba.tests.support import tag, TestCase


def matmulcore(A, B, C):
    """docstring for matmulcore"""
    m, n = A.shape
    n, p = B.shape
    for i in range(m):
        for j in range(p):
            C[i, j] = 0
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]


def axpy(a, x, y, out):
    out[0] = a * x  + y


class TestGUFunc(TestCase):
    target = 'cpu'

    def check_matmul_gufunc(self, gufunc):
        matrix_ct = 1001
        A = np.arange(matrix_ct * 2 * 4, dtype=np.float32).reshape(matrix_ct, 2, 4)
        B = np.arange(matrix_ct * 4 * 5, dtype=np.float32).reshape(matrix_ct, 4, 5)

        C = gufunc(A, B)
        Gold = ut.matrix_multiply(A, B)

        np.testing.assert_allclose(C, Gold, rtol=1e-5, atol=1e-8)

    def test_gufunc(self):
        gufunc = GUVectorize(matmulcore, '(m,n),(n,p)->(m,p)',
                             target=self.target)
        gufunc.add((float32[:, :], float32[:, :], float32[:, :]))
        gufunc = gufunc.build_ufunc()

        self.check_matmul_gufunc(gufunc)

    def test_guvectorize_decor(self):
        gufunc = guvectorize([void(float32[:,:], float32[:,:], float32[:,:])],
                             '(m,n),(n,p)->(m,p)',
                             target=self.target)(matmulcore)

        self.check_matmul_gufunc(gufunc)

    def test_ufunc_like(self):
        # Test problem that the stride of "scalar" gufunc argument not properly
        # handled when the actual argument is an array,
        # causing the same value (first value) being repeated.
        gufunc = GUVectorize(axpy, '(), (), () -> ()', target=self.target)
        gufunc.add('(intp, intp, intp, intp[:])')
        gufunc = gufunc.build_ufunc()

        x = np.arange(10, dtype=np.intp)
        out = gufunc(x, x, x)

        np.testing.assert_equal(out, x * x + x)

    def test_axis(self):
        # issue https://github.com/numba/numba/issues/6773
        @guvectorize(["f8[:],f8[:]"], "(n)->(n)")
        def my_cumsum(x, res):
            acc = 0
            for i in range(x.shape[0]):
                acc += x[i]
                res[i] = acc

        x = np.ones((20, 30))
        # Check regular call
        y = my_cumsum(x, axis=0)
        expected = np.cumsum(x, axis=0)
        np.testing.assert_equal(y, expected)
        # Check "out" kw
        out_kw = np.zeros_like(y)
        my_cumsum(x, out=out_kw, axis=0)
        np.testing.assert_equal(out_kw, expected)

    def test_docstring(self):
        @guvectorize([(int64[:], int64, int64[:])], '(n),()->(n)')
        def gufunc(x, y, res):
            "docstring for gufunc"
            for i in range(x.shape[0]):
                res[i] = x[i] + y

        self.assertEqual("numba.tests.npyufunc.test_gufunc", gufunc.__module__)
        self.assertEqual("gufunc", gufunc.__name__)
        self.assertEqual("TestGUFunc.test_docstring.<locals>.gufunc", gufunc.__qualname__)
        self.assertEqual("docstring for gufunc", gufunc.__doc__)


class TestGUFuncParallel(TestGUFunc):
    _numba_parallel_test_ = False
    target = 'parallel'


class TestDynamicGUFunc(TestCase):
    target = 'cpu'

    def test_dynamic_matmul(self):

        def check_matmul_gufunc(gufunc, A, B, C):
            Gold = ut.matrix_multiply(A, B)
            gufunc(A, B, C)
            np.testing.assert_allclose(C, Gold, rtol=1e-5, atol=1e-8)

        gufunc = GUVectorize(matmulcore, '(m,n),(n,p)->(m,p)',
                             target=self.target, is_dynamic=True)
        matrix_ct = 10
        Ai64 = np.arange(matrix_ct * 2 * 4, dtype=np.int64).reshape(matrix_ct, 2, 4)
        Bi64 = np.arange(matrix_ct * 4 * 5, dtype=np.int64).reshape(matrix_ct, 4, 5)
        Ci64 = np.arange(matrix_ct * 2 * 5, dtype=np.int64).reshape(matrix_ct, 2, 5)
        check_matmul_gufunc(gufunc, Ai64, Bi64, Ci64)

        A = np.arange(matrix_ct * 2 * 4, dtype=np.float32).reshape(matrix_ct, 2, 4)
        B = np.arange(matrix_ct * 4 * 5, dtype=np.float32).reshape(matrix_ct, 4, 5)
        C = np.arange(matrix_ct * 2 * 5, dtype=np.float32).reshape(matrix_ct, 2, 5)
        check_matmul_gufunc(gufunc, A, B, C)  # trigger compilation

        self.assertEqual(len(gufunc.types), 2)  # ensure two versions of gufunc


    def test_dynamic_ufunc_like(self):

        def check_ufunc_output(gufunc, x):
            out = np.zeros(10, dtype=x.dtype)
            out_kw = np.zeros(10, dtype=x.dtype)
            gufunc(x, x, x, out)
            gufunc(x, x, x, out=out_kw)
            golden = x * x + x
            np.testing.assert_equal(out, golden)
            np.testing.assert_equal(out_kw, golden)

        # Test problem that the stride of "scalar" gufunc argument not properly
        # handled when the actual argument is an array,
        # causing the same value (first value) being repeated.
        gufunc = GUVectorize(axpy, '(), (), () -> ()', target=self.target,
                             is_dynamic=True)
        x = np.arange(10, dtype=np.intp)
        check_ufunc_output(gufunc, x)


    def test_dynamic_scalar_output(self):
        """
        Note that scalar output is a 0-dimension array that acts as
        a pointer to the output location.
        """

        @guvectorize('(n)->()', target=self.target, nopython=True)
        def sum_row(inp, out):
            tmp = 0.
            for i in range(inp.shape[0]):
                tmp += inp[i]
            out[()] = tmp

        # inp is (10000, 3)
        # out is (10000)
        # The outter (leftmost) dimension must match or numpy broadcasting is performed.

        self.assertTrue(sum_row.is_dynamic)
        inp = np.arange(30000, dtype=np.int32).reshape(10000, 3)
        out = np.zeros(10000, dtype=np.int32)
        sum_row(inp, out)

        # verify result
        for i in range(inp.shape[0]):
            self.assertEqual(out[i], inp[i].sum())

        msg = "Too few arguments for function 'sum_row'."
        with self.assertRaisesRegex(TypeError, msg):
            sum_row(inp)

    def test_axis(self):
        # issue https://github.com/numba/numba/issues/6773
        @guvectorize("(n)->(n)")
        def my_cumsum(x, res):
            acc = 0
            for i in range(x.shape[0]):
                acc += x[i]
                res[i] = acc

        x = np.ones((20, 30))
        expected = np.cumsum(x, axis=0)
        # Check regular call
        y = np.zeros_like(expected)
        my_cumsum(x, y, axis=0)
        np.testing.assert_equal(y, expected)
        # Check "out" kw
        out_kw = np.zeros_like(y)
        my_cumsum(x, out=out_kw, axis=0)
        np.testing.assert_equal(out_kw, expected)

    def test_gufunc_attributes(self):
        @guvectorize("(n)->(n)")
        def gufunc(x, res):
            acc = 0
            for i in range(x.shape[0]):
                acc += x[i]
                res[i] = acc

        # ensure gufunc exports attributes
        attrs = ['signature', 'accumulate', 'at', 'outer', 'reduce', 'reduceat']
        for attr in attrs:
            contains = hasattr(gufunc, attr)
            self.assertTrue(contains, 'dynamic gufunc not exporting "%s"' % (attr,))

        a = np.array([1, 2, 3, 4])
        res = np.array([0, 0, 0, 0])
        gufunc(a, res)  # trigger compilation
        self.assertPreciseEqual(res, np.array([1, 3, 6, 10]))

        # other attributes are not callable from a gufunc with signature
        # see: https://github.com/numba/numba/issues/2794
        # note: this is a limitation in NumPy source code!
        self.assertEqual(gufunc.signature, "(n)->(n)")

        with self.assertRaises(RuntimeError) as raises:
            gufunc.accumulate(a)
        self.assertEqual(str(raises.exception), "Reduction not defined on ufunc with signature")

        with self.assertRaises(RuntimeError) as raises:
            gufunc.reduce(a)
        self.assertEqual(str(raises.exception), "Reduction not defined on ufunc with signature")

        with self.assertRaises(RuntimeError) as raises:
            gufunc.reduceat(a, [0, 2])
        self.assertEqual(str(raises.exception), "Reduction not defined on ufunc with signature")

        with self.assertRaises(TypeError) as raises:
            gufunc.outer(a, a)
        self.assertEqual(str(raises.exception), "method outer is not allowed in ufunc with non-trivial signature")

    def test_gufunc_attributes2(self):
        @guvectorize('(),()->()')
        def add(x, y, res):
            res[0] = x + y

        # add signature "(),() -> ()" is evaluated to None
        self.assertIsNone(add.signature)

        a = np.array([1, 2, 3, 4])
        b = np.array([4, 3, 2, 1])
        res = np.array([0, 0, 0, 0])
        add(a, b, res)  # trigger compilation
        self.assertPreciseEqual(res, np.array([5, 5, 5, 5]))

        # now test other attributes
        self.assertIsNone(add.signature)
        self.assertEqual(add.reduce(a), 10)
        self.assertPreciseEqual(add.accumulate(a), np.array([1, 3, 6, 10]))
        self.assertPreciseEqual(add.outer([0, 1], [1, 2]), np.array([[1, 2], [2, 3]]))
        self.assertPreciseEqual(add.reduceat(a, [0, 2]), np.array([3, 7]))

        x = np.array([1, 2, 3, 4])
        y = np.array([1, 2])
        add.at(x, [0, 1], y)
        self.assertPreciseEqual(x, np.array([2, 4, 3, 4]))


class TestGUVectorizeScalar(TestCase):
    """
    Nothing keeps user from out-of-bound memory access
    """
    target = 'cpu'

    def test_scalar_output(self):
        """
        Note that scalar output is a 0-dimension array that acts as
        a pointer to the output location.
        """

        @guvectorize(['void(int32[:], int32[:])'], '(n)->()',
                     target=self.target, nopython=True)
        def sum_row(inp, out):
            tmp = 0.
            for i in range(inp.shape[0]):
                tmp += inp[i]
            out[()] = tmp

        # inp is (10000, 3)
        # out is (10000)
        # The outter (leftmost) dimension must match or numpy broadcasting is performed.

        inp = np.arange(30000, dtype=np.int32).reshape(10000, 3)
        out = sum_row(inp)

        # verify result
        for i in range(inp.shape[0]):
            self.assertEqual(out[i], inp[i].sum())

    def test_scalar_input(self):

        @guvectorize(['int32[:], int32[:], int32[:]'], '(n),()->(n)',
                     target=self.target, nopython=True)
        def foo(inp, n, out):
            for i in range(inp.shape[0]):
                out[i] = inp[i] * n[0]

        inp = np.arange(3 * 10, dtype=np.int32).reshape(10, 3)
        # out = np.empty_like(inp)
        out = foo(inp, 2)

        # verify result
        self.assertPreciseEqual(inp * 2, out)

    def test_scalar_input_core_type(self):
        def pyfunc(inp, n, out):
            for i in range(inp.size):
                out[i] = n * (inp[i] + 1)

        my_gufunc = guvectorize(['int32[:], int32, int32[:]'],
                                '(n),()->(n)',
                                target=self.target)(pyfunc)

        # test single core loop execution
        arr = np.arange(10).astype(np.int32)
        got = my_gufunc(arr, 2)

        expected = np.zeros_like(got)
        pyfunc(arr, 2, expected)

        np.testing.assert_equal(got, expected)

        # test multiple core loop execution
        arr = np.arange(20).astype(np.int32).reshape(10, 2)
        got = my_gufunc(arr, 2)

        expected = np.zeros_like(got)
        for ax in range(expected.shape[0]):
            pyfunc(arr[ax], 2, expected[ax])

        np.testing.assert_equal(got, expected)

    def test_scalar_input_core_type_error(self):
        with self.assertRaises(TypeError) as raises:
            @guvectorize(['int32[:], int32, int32[:]'], '(n),(n)->(n)',
                         target=self.target)
            def pyfunc(a, b, c):
                pass
        self.assertEqual("scalar type int32 given for non scalar argument #2",
                         str(raises.exception))

    def test_ndim_mismatch(self):
        with self.assertRaises(TypeError) as raises:
            @guvectorize(['int32[:], int32[:]'], '(m,n)->(n)',
                         target=self.target)
            def pyfunc(a, b):
                pass
        self.assertEqual("type and shape signature mismatch for arg #1",
                         str(raises.exception))


class TestGUVectorizeScalarParallel(TestGUVectorizeScalar):
    _numba_parallel_test_ = False
    target = 'parallel'


class TestGUVectorizePickling(TestCase):
    def test_pickle_gufunc_non_dyanmic(self):
        """Non-dynamic gufunc.
        """
        @guvectorize(["f8,f8[:]"], "()->()")
        def double(x, out):
            out[:] = x * 2

        # pickle
        ser = pickle.dumps(double)
        cloned = pickle.loads(ser)

        # attributes carried over
        self.assertEqual(cloned._frozen, double._frozen)
        self.assertEqual(cloned.identity, double.identity)
        self.assertEqual(cloned.is_dynamic, double.is_dynamic)
        self.assertEqual(cloned.gufunc_builder._sigs,
                         double.gufunc_builder._sigs)
        # expected value of attributes
        self.assertTrue(cloned._frozen)

        cloned.disable_compile()
        self.assertTrue(cloned._frozen)

        # scalar version
        self.assertPreciseEqual(double(0.5), cloned(0.5))
        # array version
        arr = np.arange(10)
        self.assertPreciseEqual(double(arr), cloned(arr))

    def test_pickle_gufunc_dyanmic_null_init(self):
        """Dynamic gufunc w/o prepopulating before pickling.
        """
        @guvectorize("()->()", identity=1)
        def double(x, out):
            out[:] = x * 2

        # pickle
        ser = pickle.dumps(double)
        cloned = pickle.loads(ser)

        # attributes carried over
        self.assertEqual(cloned._frozen, double._frozen)
        self.assertEqual(cloned.identity, double.identity)
        self.assertEqual(cloned.is_dynamic, double.is_dynamic)
        self.assertEqual(cloned.gufunc_builder._sigs,
                         double.gufunc_builder._sigs)
        # expected value of attributes
        self.assertFalse(cloned._frozen)

        # scalar version
        expect = np.zeros(1)
        got = np.zeros(1)
        double(0.5, out=expect)
        cloned(0.5, out=got)
        self.assertPreciseEqual(expect, got)
        # array version
        arr = np.arange(10)
        expect = np.zeros_like(arr)
        got = np.zeros_like(arr)
        double(arr, out=expect)
        cloned(arr, out=got)
        self.assertPreciseEqual(expect, got)

    def test_pickle_gufunc_dynamic_initialized(self):
        """Dynamic gufunc prepopulated before pickling.

        Once unpickled, we disable compilation to verify that the gufunc
        compilation state is carried over.
        """
        @guvectorize("()->()", identity=1)
        def double(x, out):
            out[:] = x * 2

        # prepopulate scalar
        expect = np.zeros(1)
        got = np.zeros(1)
        double(0.5, out=expect)
        # prepopulate array
        arr = np.arange(10)
        expect = np.zeros_like(arr)
        got = np.zeros_like(arr)
        double(arr, out=expect)

        # pickle
        ser = pickle.dumps(double)
        cloned = pickle.loads(ser)

        # attributes carried over
        self.assertEqual(cloned._frozen, double._frozen)
        self.assertEqual(cloned.identity, double.identity)
        self.assertEqual(cloned.is_dynamic, double.is_dynamic)
        self.assertEqual(cloned.gufunc_builder._sigs,
                         double.gufunc_builder._sigs)
        # expected value of attributes
        self.assertFalse(cloned._frozen)

        # disable compilation
        cloned.disable_compile()
        self.assertTrue(cloned._frozen)
        # scalar version
        expect = np.zeros(1)
        got = np.zeros(1)
        double(0.5, out=expect)
        cloned(0.5, out=got)
        self.assertPreciseEqual(expect, got)
        # array version
        expect = np.zeros_like(arr)
        got = np.zeros_like(arr)
        double(arr, out=expect)
        cloned(arr, out=got)
        self.assertPreciseEqual(expect, got)


if __name__ == '__main__':
    unittest.main()
