from numba import cuda
from numba.core.extending import overload
from numba.cuda.testing import CUDATestCase, skip_on_cudasim, unittest

import numpy as np


# Dummy function definitions to overload

def generic_func_1():
    pass


def cuda_func_1():
    pass


def generic_func_2():
    pass


def cuda_func_2():
    pass


def generic_calls_generic():
    pass


def generic_calls_cuda():
    pass


def cuda_calls_generic():
    pass


def cuda_calls_cuda():
    pass


def hardware_overloaded():
    pass


def generic_calls_hardware_overloaded():
    pass


def cuda_calls_hardware_overloaded():
    pass


def hardware_overloaded_calls_hardware_overloaded():
    pass


# To recognise which functions are resolved for a call, we identify each with a
# prime number. Each function called multiplies a value by its prime (starting
# with the value 1), and we can check that the result is as expected based on
# the final value after all multiplications.

GENERIC_FUNCTION_1 = 2
CUDA_FUNCTION_1 = 3
GENERIC_FUNCTION_2 = 5
CUDA_FUNCTION_2 = 7
GENERIC_CALLS_GENERIC = 11
GENERIC_CALLS_CUDA = 13
CUDA_CALLS_GENERIC = 17
CUDA_CALLS_CUDA = 19
GENERIC_HARDWARE_OL = 23
CUDA_HARDWARE_OL = 29
GENERIC_CALLS_HARDWARE_OL = 31
CUDA_CALLS_HARDWARE_OL = 37
GENERIC_HARDWARE_OL_CALLS_HARDWARE_OL = 41
CUDA_HARDWARE_OL_CALLS_HARDWARE_OL = 43


# Overload implementations

@overload(generic_func_1, hardware='generic')
def ol_generic_func_1(x):
    def impl(x):
        x[0] *= GENERIC_FUNCTION_1
    return impl


@overload(cuda_func_1, hardware='cuda')
def ol_cuda_func_1(x):
    def impl(x):
        x[0] *= CUDA_FUNCTION_1
    return impl


@overload(generic_func_2, hardware='generic')
def ol_generic_func_2(x):
    def impl(x):
        x[0] *= GENERIC_FUNCTION_2
    return impl


@overload(cuda_func_2, hardware='cuda')
def ol_cuda_func(x):
    def impl(x):
        x[0] *= CUDA_FUNCTION_2
    return impl


@overload(generic_calls_generic, hardware='generic')
def ol_generic_calls_generic(x):
    def impl(x):
        x[0] *= GENERIC_CALLS_GENERIC
        generic_func_1(x)
    return impl


@overload(generic_calls_cuda, hardware='generic')
def ol_generic_calls_cuda(x):
    def impl(x):
        x[0] *= GENERIC_CALLS_CUDA
        cuda_func_1(x)
    return impl


@overload(cuda_calls_generic, hardware='cuda')
def ol_cuda_calls_generic(x):
    def impl(x):
        x[0] *= CUDA_CALLS_GENERIC
        generic_func_1(x)
    return impl


@overload(cuda_calls_cuda, hardware='cuda')
def ol_cuda_calls_cuda(x):
    def impl(x):
        x[0] *= CUDA_CALLS_CUDA
        cuda_func_1(x)
    return impl


@overload(hardware_overloaded, hardware='generic')
def ol_hardware_overloaded_generic(x):
    def impl(x):
        x[0] *= GENERIC_HARDWARE_OL
    return impl


@overload(hardware_overloaded, hardware='cuda')
def ol_hardware_overloaded_cuda(x):
    def impl(x):
        x[0] *= CUDA_HARDWARE_OL
    return impl


@overload(generic_calls_hardware_overloaded, hardware='generic')
def ol_generic_calls_hardware_overloaded(x):
    def impl(x):
        x[0] *= GENERIC_CALLS_HARDWARE_OL
        hardware_overloaded(x)
    return impl


@overload(cuda_calls_hardware_overloaded, hardware='cuda')
def ol_cuda_calls_hardware_overloaded(x):
    def impl(x):
        x[0] *= CUDA_CALLS_HARDWARE_OL
        hardware_overloaded(x)
    return impl


@overload(hardware_overloaded_calls_hardware_overloaded, hardware='generic')
def ol_generic_calls_hardware_overloaded_generic(x):
    def impl(x):
        x[0] *= GENERIC_HARDWARE_OL_CALLS_HARDWARE_OL
        hardware_overloaded(x)
    return impl


@overload(hardware_overloaded_calls_hardware_overloaded, hardware='cuda')
def ol_generic_calls_hardware_overloaded_cuda(x):
    def impl(x):
        x[0] *= CUDA_HARDWARE_OL_CALLS_HARDWARE_OL
        hardware_overloaded(x)
    return impl


@skip_on_cudasim('Overloading not supported in cudasim')
class TestOverload(CUDATestCase):
    def check_overload(self, kernel, expected):
        x = np.ones(1, dtype=np.int32)
        cuda.jit(kernel)[1, 1](x)
        self.assertEqual(x[0], expected)

    def test_generic(self):
        def kernel(x):
            generic_func_1(x)

        expected = GENERIC_FUNCTION_1
        self.check_overload(kernel, expected)

    def test_cuda(self):
        def kernel(x):
            cuda_func_1(x)

        expected = CUDA_FUNCTION_1
        self.check_overload(kernel, expected)

    def test_generic_and_cuda(self):
        def kernel(x):
            generic_func_1(x)
            cuda_func_1(x)

        expected = GENERIC_FUNCTION_1 * CUDA_FUNCTION_1
        self.check_overload(kernel, expected)

    def test_call_two_generic_calls(self):
        def kernel(x):
            generic_func_1(x)
            generic_func_2(x)

        expected = GENERIC_FUNCTION_1 * GENERIC_FUNCTION_2
        self.check_overload(kernel, expected)

    def test_call_two_cuda_calls(self):
        def kernel(x):
            cuda_func_1(x)
            cuda_func_2(x)

        expected = CUDA_FUNCTION_1 * CUDA_FUNCTION_2
        self.check_overload(kernel, expected)

    def test_generic_calls_generic(self):
        def kernel(x):
            generic_calls_generic(x)

        expected = GENERIC_CALLS_GENERIC * GENERIC_FUNCTION_1
        self.check_overload(kernel, expected)

    def test_generic_calls_cuda(self):
        def kernel(x):
            generic_calls_cuda(x)

        expected = GENERIC_CALLS_CUDA * CUDA_FUNCTION_1
        self.check_overload(kernel, expected)

    def test_cuda_calls_generic(self):
        def kernel(x):
            cuda_calls_generic(x)

        expected = CUDA_CALLS_GENERIC * GENERIC_FUNCTION_1
        self.check_overload(kernel, expected)

    def test_cuda_calls_cuda(self):
        def kernel(x):
            cuda_calls_cuda(x)

        expected = CUDA_CALLS_CUDA * CUDA_FUNCTION_1
        self.check_overload(kernel, expected)

    def test_call_hardware_overloaded(self):
        def kernel(x):
            hardware_overloaded(x)

        expected = CUDA_HARDWARE_OL
        self.check_overload(kernel, expected)

    def test_generic_calls_hardware_overloaded(self):
        def kernel(x):
            generic_calls_hardware_overloaded(x)

        expected = GENERIC_CALLS_HARDWARE_OL * CUDA_HARDWARE_OL
        self.check_overload(kernel, expected)

    def test_cuda_calls_hardware_overloaded(self):
        def kernel(x):
            cuda_calls_hardware_overloaded(x)

        expected = CUDA_CALLS_HARDWARE_OL * CUDA_HARDWARE_OL
        self.check_overload(kernel, expected)

    def test_hardware_overloaded_calls_hardware_overloaded(self):
        def kernel(x):
            hardware_overloaded_calls_hardware_overloaded(x)

        expected = CUDA_HARDWARE_OL_CALLS_HARDWARE_OL * CUDA_HARDWARE_OL
        self.check_overload(kernel, expected)


if __name__ == '__main__':
    unittest.main()
