import numpy as np
import sys

from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba.core.config import ENABLE_CUDASIM

CONST_EMPTY = np.array([])
CONST1D = np.arange(10, dtype=np.float64) / 2.
CONST2D = np.asfortranarray(
    np.arange(100, dtype=np.int32).reshape(10, 10))
CONST3D = ((np.arange(5 * 5 * 5, dtype=np.complex64).reshape(5, 5, 5) + 1j) /
           2j)
CONST3BYTES = np.arange(3, dtype=np.uint8)

CONST_RECORD_EMPTY = np.array(
    [],
    dtype=[('x', float), ('y', int)])
CONST_RECORD = np.array(
    [(1.0, 2), (3.0, 4)],
    dtype=[('x', float), ('y', int)])
CONST_RECORD_ALIGN = np.array(
    [(1, 2, 3, 0xDEADBEEF, 8), (4, 5, 6, 0xBEEFDEAD, 10)],
    dtype=np.dtype(
        dtype=[
            ('a', np.uint8),
            ('b', np.uint8),
            ('x', np.uint8),
            ('y', np.uint32),
            ('z', np.uint8),
        ],
        align=True))


def cuconstEmpty(A):
    C = cuda.const.array_like(CONST_EMPTY)
    i = cuda.grid(1)
    A[i] = len(C)


def cuconst(A):
    C = cuda.const.array_like(CONST1D)
    i = cuda.grid(1)

    # +1 or it'll be loaded & stored as a u32
    A[i] = C[i] + 1.0


def cuconst2d(A):
    C = cuda.const.array_like(CONST2D)
    i, j = cuda.grid(2)
    A[i, j] = C[i, j]


def cuconst3d(A):
    C = cuda.const.array_like(CONST3D)
    i = cuda.threadIdx.x
    j = cuda.threadIdx.y
    k = cuda.threadIdx.z
    A[i, j, k] = C[i, j, k]


def cuconstRecEmpty(A):
    C = cuda.const.array_like(CONST_RECORD_EMPTY)
    i = cuda.grid(1)
    A[i] = len(C)


def cuconstRec(A, B):
    C = cuda.const.array_like(CONST_RECORD)
    i = cuda.grid(1)
    A[i] = C[i]['x']
    B[i] = C[i]['y']


def cuconstRecAlign(A, B, C, D, E):
    Z = cuda.const.array_like(CONST_RECORD_ALIGN)
    i = cuda.grid(1)
    A[i] = Z[i]['a']
    B[i] = Z[i]['b']
    C[i] = Z[i]['x']
    D[i] = Z[i]['y']
    E[i] = Z[i]['z']


def cuconstAlign(z):
    a = cuda.const.array_like(CONST3BYTES)
    b = cuda.const.array_like(CONST1D)
    i = cuda.grid(1)
    z[i] = a[i] + b[i]


class TestCudaConstantMemory(CUDATestCase):
    def test_const_array(self):
        jcuconst = cuda.jit('void(float64[:])')(cuconst)
        A = np.zeros_like(CONST1D)
        jcuconst[2, 5](A)
        self.assertTrue(np.all(A == CONST1D + 1))

        if not ENABLE_CUDASIM:
            self.assertIn(
                'ld.const.f64',
                jcuconst.ptx,
                "as we're adding to it, load as a double")

    def test_const_empty(self):
        jcuconstEmpty = cuda.jit('void(int64[:])')(cuconstEmpty)
        A = np.full(1, fill_value=-1, dtype=np.int64)
        jcuconstEmpty[1, 1](A)
        self.assertTrue(np.all(A == 0))

    def test_const_align(self):
        jcuconstAlign = cuda.jit('void(float64[:])')(cuconstAlign)
        A = np.full(3, fill_value=np.nan, dtype=float)
        jcuconstAlign[1, 3](A)
        self.assertTrue(np.all(A == (CONST3BYTES + CONST1D[:3])))

    def test_const_array_2d(self):
        jcuconst2d = cuda.jit('void(int32[:,:])')(cuconst2d)
        A = np.zeros_like(CONST2D, order='C')
        jcuconst2d[(2, 2), (5, 5)](A)
        self.assertTrue(np.all(A == CONST2D))

        if not ENABLE_CUDASIM:
            self.assertIn(
                'ld.const.u32',
                jcuconst2d.ptx,
                "load the ints as ints")

    def test_const_array_3d(self):
        jcuconst3d = cuda.jit('void(complex64[:,:,:])')(cuconst3d)
        A = np.zeros_like(CONST3D, order='F')
        jcuconst3d[1, (5, 5, 5)](A)
        self.assertTrue(np.all(A == CONST3D))

        if not ENABLE_CUDASIM:
            # CUDA 9.2 - 11.1 use two f32 loads to load the complex. CUDA < 9.2
            # and > 11.1 use a vector of 2x f32. The root cause of these
            # codegen differences is not known, but must be accounted for in
            # this test.
            if cuda.runtime.get_version() in ((9, 0), (9, 1), (11, 2)):
                complex_load = 'ld.const.v2.f32'
                description = 'Load the complex as a vector of 2x f32'
            else:
                complex_load = 'ld.const.f32'
                description = 'load each half of the complex as f32'

            self.assertIn(complex_load, jcuconst3d.ptx, description)

    def test_const_record_empty(self):
        jcuconstRecEmpty = cuda.jit('void(int64[:])')(cuconstRecEmpty)
        A = np.full(1, fill_value=-1, dtype=np.int64)
        jcuconstRecEmpty[1, 1](A)
        self.assertTrue(np.all(A == 0))

    def test_const_record(self):
        A = np.zeros(2, dtype=float)
        B = np.zeros(2, dtype=int)
        jcuconst = cuda.jit(cuconstRec).specialize(A, B)

        jcuconst[2, 1](A, B)
        np.testing.assert_allclose(A, CONST_RECORD['x'])
        np.testing.assert_allclose(B, CONST_RECORD['y'])

    @skip_on_cudasim('PTX inspection not supported on the simulator')
    def test_const_record_optimization(self):
        A = np.zeros(2, dtype=float)
        B = np.zeros(2, dtype=int)
        jcuconst = cuda.jit(cuconstRec).specialize(A, B)

        rtver = cuda.runtime.get_version()
        old_runtime = rtver in ((9, 0), (9, 1))
        nvvm70_runtime = rtver >= (11, 2)
        windows = sys.platform.startswith('win')

        if old_runtime:
            if windows:
                # For some reason Win64 / CUDA 9.1 and 9.2 decide to do two u32
                # loads, and shifts and ors the values to get the float `x`
                # field, then uses another ld.const.u32 to load the int `y` as
                # a 32-bit value!
                self.assertIn('ld.const.u32', jcuconst.ptx,
                              'load record fields as u32')
            else:
                # Load of the x and y fields fused into a single instruction
                self.assertIn('ld.const.v2.f64', jcuconst.ptx,
                              'load record fields as vector of 2x f64')
        elif nvvm70_runtime:
            if windows:
                # Two ld.const.u32 as above, but using a bit-field insert to
                # combine them
                self.assertIn('ld.const.u32', jcuconst.ptx,
                              'load record fields as u32')
            else:
                # Load of the x and y fields fused into a single instruction
                self.assertIn('ld.const.v2.u64', jcuconst.ptx,
                              'load record fields as vector of 2x u64')
        else:
            # In newer toolkits, constant values are all loaded 8 bits at a
            # time. Check that there are enough 8-bit loads for everything to
            # have been loaded. This is possibly less than optimal, but is the
            # observed behaviour with current toolkit versions when IR is not
            # optimized before sending to NVVM.
            u8_load_count = len([s for s in jcuconst.ptx.split()
                                 if 'ld.const.u8' in s])

            if windows:
                # NumPy ints are 32-bit on Windows by default, so only 4 bytes
                # for loading the int (and 8 for the float)
                expected_load_count = 12
            else:
                # int is 64-bit elsewhere
                expected_load_count = 16
            self.assertGreaterEqual(u8_load_count, expected_load_count,
                                    'load record values as individual bytes')

    def test_const_record_align(self):
        A = np.zeros(2, dtype=np.float64)
        B = np.zeros(2, dtype=np.float64)
        C = np.zeros(2, dtype=np.float64)
        D = np.zeros(2, dtype=np.float64)
        E = np.zeros(2, dtype=np.float64)
        jcuconst = cuda.jit(cuconstRecAlign).specialize(A, B, C, D, E)

        jcuconst[2, 1](A, B, C, D, E)
        np.testing.assert_allclose(A, CONST_RECORD_ALIGN['a'])
        np.testing.assert_allclose(B, CONST_RECORD_ALIGN['b'])
        np.testing.assert_allclose(C, CONST_RECORD_ALIGN['x'])
        np.testing.assert_allclose(D, CONST_RECORD_ALIGN['y'])
        np.testing.assert_allclose(E, CONST_RECORD_ALIGN['z'])

    @skip_on_cudasim('PTX inspection not supported on the simulator')
    def test_const_record_align_optimization(self):
        rtver = cuda.runtime.get_version()

        A = np.zeros(2, dtype=np.float64)
        B = np.zeros(2, dtype=np.float64)
        C = np.zeros(2, dtype=np.float64)
        D = np.zeros(2, dtype=np.float64)
        E = np.zeros(2, dtype=np.float64)
        jcuconst = cuda.jit(cuconstRecAlign).specialize(A, B, C, D, E)

        if rtver >= (10, 2) and rtver <= (11, 1):
            # Code generation differs slightly in 10.2 - 11.1 - the first
            # bytes are loaded as individual bytes, so we'll check that
            # ld.const.u8 occurs at least four times (the first three bytes,
            # then the last byte by itself)
            msg = 'load first three bytes and last byte individually'
            u8_load_count = len([s for s in jcuconst.ptx.split()
                                 if 'ld.const.u8' in s])
            self.assertGreaterEqual(u8_load_count, 4, msg)
        else:
            # On earlier versions, a vector of 4 bytes is used to load the
            # first three bytes.
            first_bytes = 'ld.const.v4.u8'
            first_bytes_msg = 'load the first three bytes as a vector'

            self.assertIn(
                first_bytes,
                jcuconst.ptx,
                first_bytes_msg)

        self.assertIn(
            'ld.const.u32',
            jcuconst.ptx,
            'load the uint32 natively')

        # On 10.2 and above, we already checked for loading the last byte by
        # itself - no need to repeat the check.
        if rtver < (10, 2):
            self.assertIn(
                'ld.const.u8',
                jcuconst.ptx,
                'load the last byte by itself')


if __name__ == '__main__':
    unittest.main()
