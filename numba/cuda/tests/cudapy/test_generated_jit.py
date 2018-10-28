from __future__ import print_function, absolute_import, division
from functools import wraps
import numpy as np
from numba import cuda, generated_jit
from numba.cuda.testing import unittest, SerialMixin
from numba.cuda.testing import skip_on_cudasim


class TestCudaGeneratedJit(SerialMixin, unittest.TestCase):
    def test_generated_jit(self):

        @generated_jit(target='cuda')
        def kernel(array, out):
            TPB = 16
            dtype = array.dtype

            def _kernel(array, out):
                shared = cuda.shared.array(TPB, dtype)

                i = cuda.grid(1)

                if i >= array.shape[0]:
                    return

                out[i] = array[i] + 1

            return _kernel


        array = np.zeros(200, dtype=np.int32)
        out = cuda.device_array(array.shape, dtype=array.dtype)
        kernel[(1,1,1), (200,1,1)](array, out)

        host_out = out.copy_to_host()

        assert host_out.dtype == array.dtype
        assert np.all(host_out == 1)


        array = np.zeros(200, dtype=np.float64)
        out = cuda.device_array(array.shape, dtype=array.dtype)
        kernel[(1,1,1), (200,1,1)](array, out)

        host_out = out.copy_to_host()

        assert host_out.dtype == array.dtype
        assert np.all(host_out == 1)


if __name__ == '__main__':
    unittest.main()

