import multiprocessing as mp

from numba import cuda
from numba.cuda.cudadrv.driver import CudaAPIError, driver
from numba.cuda.cudadrv.error import CudaSupportError
from numba.cuda.testing import unittest, CUDATestCase


# A mock of cuInit that always raises a CudaAPIError
def cuInit_raising(arg):
    raise CudaAPIError(999, 'CUDA_ERROR_UNKNOWN')


# Test code to run in a child that patches driver.cuInit to a variant that
# always raises. We can't use mock.patch.object here because driver.cuInit is
# not assigned until we attempt to initialize - mock.patch.object cannot locate
# the non-existent original method, and so fails. Instead we patch
# driver.cuInit with our raising version prior to any attempt to initialize.
def cuInit_raising_test(result_queue):
    driver.cuInit = cuInit_raising

    success = False
    msg = None

    try:
        # A CUDA operation that forces initialization of the device
        cuda.device_array(1)
    except CudaSupportError as e:
        success = True
        msg = e.msg

    result_queue.put((success, msg))


# Similar to cuInit_raising_test above, but for testing that the string
# returned by cuda_error() is as expected.
def initialization_error_test(result_queue):
    driver.cuInit = cuInit_raising

    success = False
    msg = None

    try:
        # A CUDA operation that forces initialization of the device
        cuda.device_array(1)
    except CudaSupportError:
        success = True
        msg = cuda.cuda_error()

    result_queue.put((success, msg))


class TestInit(CUDATestCase):
    def _test_init_failure(self, target, expected):
        # Run the initialization failure test in a separate subprocess
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        proc = ctx.Process(target=target, args=(result_queue,))
        proc.start()
        proc.join()
        success, msg = result_queue.get()

        # Ensure the child process raised an except during initialization
        # before checking the message
        if not success:
            self.fail('CudaSupportError not raised')

        self.assertEqual(msg, expected)

    def test_init_failure_raising(self):
        expected = 'Error at driver init: CUDA_ERROR_UNKNOWN (999)'
        self._test_init_failure(cuInit_raising_test, expected)

    def test_init_failure_error(self):
        expected = 'CUDA_ERROR_UNKNOWN (999)'
        self._test_init_failure(initialization_error_test, expected)

    def test_init_success(self):
        # Here we assume that initialization is successful (because many bad
        # things will happen with the test suite if it is not) and check that
        # there is no error recorded.
        self.assertIsNone(cuda.cuda_error())


if __name__ == '__main__':
    unittest.main()
