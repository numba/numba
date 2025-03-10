import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase


def foo(inp, out):
    for i in range(out.shape[0]):
        out[i] = inp[i]


def copy(inp, out):
    i = cuda.grid(1)
    cufoo(inp[i, :], out[i, :])


class TestCudaSlicing(CUDATestCase):
    def test_slice_as_arg(self):
        global cufoo
        cufoo = cuda.jit("void(int32[:], int32[:])", device=True)(foo)
        cucopy = cuda.jit("void(int32[:,:], int32[:,:])")(copy)

        inp = np.arange(100, dtype=np.int32).reshape(10, 10)
        out = np.zeros_like(inp)

        cucopy[1, 10](inp, out)

    def test_assign_empty_slice(self):
        # Issue #5017. Assigning to an empty slice should not result in a
        # CudaAPIError.
        N = 0
        a = range(N)
        arr = cuda.device_array(len(a))
        arr[:] = cuda.to_device(a)

    def test_slice_error_handling_codegen(self):
        # This checks that the error handling code for invalid slice assignment
        # will compile for the CUDA target. There is nothing to run or check
        # because the CUDA target cannot propagate the raised exception across
        # the (generated) function call boundary, in essence it fails silently.
        # Further the built-in CUDA implementation does not support a "dynamic"
        # sequence type (i.e. list or set) as it has no NRT available. As a
        # result it's not possible at runime to take the execution path for
        # raising the exception coming from the "sequence" side of the
        # "mismatched" set-slice operation code generation. This is because it
        # is preempted by an exception raised from the tuple being "seen" as the
        # wrong size earlier in the execution.
        # See #9906 for context.

        # Compile the "assign slice from sequence" path
        @cuda.jit("void(f4[:, :, :], i4, i4)")
        def check_sequence_setslice(tmp, a, b):
            tmp[a, b] = 1, 1, 1

        # Compile the "assign slice from array" path
        @cuda.jit("void(f4[:, :, :], f4[:], i4, i4)")
        def check_array_setslice(tmp, value, a, b):
            tmp[a, b] = value


if __name__ == '__main__':
    unittest.main()
