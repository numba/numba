from __future__ import print_function, absolute_import, division
from functools import wraps
import numpy as np
from numba import cuda, generated_jit
from numba.cuda.testing import unittest, SerialMixin
from numba.cuda.testing import skip_on_cudasim


class TestCudaGeneratedJit(SerialMixin, unittest.TestCase):
    def test_generated_jit(self):
        """ Basic generated_jit test case """
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
        kernel[(1, 1, 1), (200, 1, 1)](array, out)

        host_out = out.copy_to_host()

        assert host_out.dtype == array.dtype
        assert np.all(host_out == 1)

        array = np.zeros(200, dtype=np.float64)
        out = cuda.device_array(array.shape, dtype=array.dtype)
        kernel[(1, 1, 1), (200, 1, 1)](array, out)

        host_out = out.copy_to_host()

        assert host_out.dtype == array.dtype
        assert np.all(host_out == 1)

    def test_generated_jit_device(self):
        """ Can compile device functions into generated_jit kernels """
        @generated_jit(target='cuda')
        def kernel(array):
            @cuda.jit(device=True)
            def device_add(array):
                i = cuda.grid(1)

                if i >= array.shape[0]:
                    return

                array[i] += 1

            @wraps(kernel)
            def _kernel(array):
                device_add(array)

            return _kernel

        array = np.zeros(200, dtype=np.float64)
        array = cuda.to_device(array)
        kernel[(1, 1, 1), (200, 1, 1)](array)

        host_out = array.copy_to_host()

        assert host_out.dtype == array.dtype
        assert np.all(host_out == 1)

    def test_generated_jit_device_fails(self):
        """ Can't use generated_jit with device functions """
        @generated_jit(device=True, target='cuda')
        def device_add(array):
            i = cuda.grid(1)

            if i >= array.shape[0]:
                return

            array[i] += 1

        @cuda.jit
        def kernel(array):
            device_add(array)

        array = np.zeros(200, dtype=np.float64)
        array = cuda.to_device(array)

        ex_regex = ("Only direct implementations "
                    "are supported for device functions")

        with self.assertRaisesRegexp(NotImplementedError, ex_regex):
            kernel[(1, 1, 1), (200, 1, 1)](array)


if __name__ == '__main__':
    unittest.main()
