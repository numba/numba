import numba
import numpy as np
from numba import dppl, njit
from numba.core import errors
from numba.tests.support import captured_stdout
from numba.dppl.testing import DPPLTestCase, unittest
import dpctl


class TestWithDPPLContext(DPPLTestCase):

    @unittest.skipIf(not dpctl.has_gpu_queues(), "No GPU platforms available")
    def test_with_dppl_context_gpu(self):

        @njit
        def nested_func(a, b):
            np.sin(a, b)

        @njit
        def func(b):
            a = np.ones((64), dtype=np.float64)
            nested_func(a, b)

        numba.dppl.compiler.DEBUG = 1
        expected = np.ones((64), dtype=np.float64)
        got_gpu = np.ones((64), dtype=np.float64)

        with captured_stdout() as got_gpu_message:
            with dpctl.device_context("opencl:gpu"):
                func(got_gpu)

        numba.dppl.compiler.DEBUG = 0
        func(expected)

        np.testing.assert_array_equal(expected, got_gpu)
        self.assertTrue('Parfor lowered on DPPL-device' in got_gpu_message.getvalue())

    @unittest.skipIf(not dpctl.has_cpu_queues(), "No CPU platforms available")
    def test_with_dppl_context_cpu(self):

        @njit
        def nested_func(a, b):
            np.sin(a, b)

        @njit
        def func(b):
            a = np.ones((64), dtype=np.float64)
            nested_func(a, b)

        numba.dppl.compiler.DEBUG = 1
        expected = np.ones((64), dtype=np.float64)
        got_cpu = np.ones((64), dtype=np.float64)

        with captured_stdout() as got_cpu_message:
            with dpctl.device_context("opencl:cpu"):
                func(got_cpu)

        numba.dppl.compiler.DEBUG = 0
        func(expected)

        np.testing.assert_array_equal(expected, got_cpu)
        self.assertTrue('Parfor lowered on DPPL-device' in got_cpu_message.getvalue())


    @unittest.skipIf(not dpctl.has_gpu_queues(), "No GPU platforms available")
    def test_with_dppl_context_target(self):

        @njit(target='cpu')
        def nested_func_target(a, b):
            np.sin(a, b)

        @njit(target='gpu')
        def func_target(b):
            a = np.ones((64), dtype=np.float64)
            nested_func_target(a, b)

        @njit
        def func_no_target(b):
            a = np.ones((64), dtype=np.float64)
            nested_func_target(a, b)

        @njit(parallel=False)
        def func_no_parallel(b):
            a = np.ones((64), dtype=np.float64)
            return a


        a = np.ones((64), dtype=np.float64)
        b = np.ones((64), dtype=np.float64)

        with self.assertRaises(errors.UnsupportedError) as raises_1:
            with dpctl.device_context("opencl:gpu"):
                nested_func_target(a, b)

        with self.assertRaises(errors.UnsupportedError) as raises_2:
            with dpctl.device_context("opencl:gpu"):
                func_target(a)

        with self.assertRaises(errors.UnsupportedError) as raises_3:
            with dpctl.device_context("opencl:gpu"):
                func_no_target(a)

        with self.assertRaises(errors.UnsupportedError) as raises_4:
            with dpctl.device_context("opencl:gpu"):
                func_no_parallel(a)

        msg_1 = "Can't use 'with' context with explicitly specified target"
        msg_2 = "Can't use 'with' context with parallel option"
        self.assertTrue(msg_1 in str(raises_1.exception))
        self.assertTrue(msg_1 in str(raises_2.exception))
        self.assertTrue(msg_1 in str(raises_3.exception))
        self.assertTrue(msg_2 in str(raises_4.exception))


if __name__ == '__main__':
    unittest.main()
