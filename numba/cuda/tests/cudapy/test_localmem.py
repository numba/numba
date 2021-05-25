import numpy as np

from numba import cuda, int32, complex128, void
from numba.core.errors import TypingError
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim


def culocal(A, B):
    C = cuda.local.array(1000, dtype=int32)
    for i in range(C.shape[0]):
        C[i] = A[i]
    for i in range(C.shape[0]):
        B[i] = C[i]


def culocalcomplex(A, B):
    C = cuda.local.array(100, dtype=complex128)
    for i in range(C.shape[0]):
        C[i] = A[i]
    for i in range(C.shape[0]):
        B[i] = C[i]


def culocal1tuple(A, B):
    C = cuda.local.array((5,), dtype=int32)
    for i in range(C.shape[0]):
        C[i] = A[i]
    for i in range(C.shape[0]):
        B[i] = C[i]


@skip_on_cudasim('PTX inspection not available in cudasim')
class TestCudaLocalMem(CUDATestCase):
    def test_local_array(self):
        sig = (int32[:], int32[:])
        jculocal = cuda.jit(sig)(culocal)
        self.assertTrue('.local' in jculocal.ptx[sig])
        A = np.arange(1000, dtype='int32')
        B = np.zeros_like(A)
        jculocal[1, 1](A, B)
        self.assertTrue(np.all(A == B))

    def test_local_array_1_tuple(self):
        """Ensure that local arrays can be constructed with 1-tuple shape
        """
        jculocal = cuda.jit('void(int32[:], int32[:])')(culocal1tuple)
        # Don't check if .local is in the ptx because the optimizer
        # may reduce it to registers.
        A = np.arange(5, dtype='int32')
        B = np.zeros_like(A)
        jculocal[1, 1](A, B)
        self.assertTrue(np.all(A == B))

    def test_local_array_complex(self):
        sig = 'void(complex128[:], complex128[:])'
        jculocalcomplex = cuda.jit(sig)(culocalcomplex)
        # The local memory would be turned into register
        # self.assertTrue('.local' in jculocalcomplex.ptx)
        A = (np.arange(100, dtype='complex128') - 1) / 2j
        B = np.zeros_like(A)
        jculocalcomplex[1, 1](A, B)
        self.assertTrue(np.all(A == B))

    def check_dtype(self, f):
        # Find the typing of the dtype argument to cuda.local.array
        annotation = next(iter(f.overloads.values()))._type_annotation
        l_dtype = annotation.typemap['l'].dtype
        # Ensure that the typing is correct
        self.assertEqual(l_dtype, int32)

    @skip_on_cudasim("Can't check typing in simulator")
    def test_numba_dtype(self):
        # Check that Numba types can be used as the dtype of a local array
        @cuda.jit(void(int32[::1]))
        def f(x):
            l = cuda.local.array(10, dtype=int32)
            l[0] = x[0]
            x[0] = l[0]

        self.check_dtype(f)

    @skip_on_cudasim("Can't check typing in simulator")
    def test_numpy_dtype(self):
        # Check that NumPy types can be used as the dtype of a local array
        @cuda.jit(void(int32[::1]))
        def f(x):
            l = cuda.local.array(10, dtype=np.int32)
            l[0] = x[0]
            x[0] = l[0]

        self.check_dtype(f)

    @skip_on_cudasim("Can't check typing in simulator")
    def test_string_dtype(self):
        # Check that strings can be used to specify the dtype of a local array
        @cuda.jit(void(int32[::1]))
        def f(x):
            l = cuda.local.array(10, dtype='int32')
            l[0] = x[0]
            x[0] = l[0]

        self.check_dtype(f)

    @skip_on_cudasim("Can't check typing in simulator")
    def test_invalid_string_dtype(self):
        # Check that strings of invalid dtypes cause a typing error
        re = ".*Invalid NumPy dtype specified: 'int33'.*"
        with self.assertRaisesRegex(TypingError, re):
            @cuda.jit(void(int32[::1]))
            def f(x):
                l = cuda.local.array(10, dtype='int33')
                l[0] = x[0]
                x[0] = l[0]


if __name__ == '__main__':
    unittest.main()
