import numpy as np
from numba.cuda.testing import unittest, CUDATestCase
from numba import cuda
from numba.core import config


class MyError(Exception):
    pass


regex_pattern = (
    r'In function [\'"]test_exc[\'"], file [\:\.\/\\\-a-zA-Z_0-9]+, line \d+'
)


@cuda.jit(device=True)
def device_function(x):
    if x == 1:
        raise ValueError('Error')
    elif x == 2:
        raise TypeError('Error')
    elif x == 3:
        raise RuntimeError('Error')
    elif x == 4:
        assert False, 'Error'


class TestUserExc(CUDATestCase):

    def test_user_exception(self):
        @cuda.jit("void(int32)", debug=True)
        def test_exc(x):
            if x == 1:
                raise MyError
            elif x == 2:
                raise MyError("foo")

        test_exc[1, 1](0)    # no raise
        with self.assertRaises(MyError) as cm:
            test_exc[1, 1](1)
        if not config.ENABLE_CUDASIM:
            self.assertRegexpMatches(str(cm.exception), regex_pattern)
        self.assertIn("tid=[0, 0, 0] ctaid=[0, 0, 0]", str(cm.exception))
        with self.assertRaises(MyError) as cm:
            test_exc[1, 1](2)
        if not config.ENABLE_CUDASIM:
            self.assertRegexpMatches(str(cm.exception), regex_pattern)
            self.assertRegexpMatches(str(cm.exception), regex_pattern)
        self.assertIn("tid=[0, 0, 0] ctaid=[0, 0, 0]: foo", str(cm.exception))


@cuda.jit(debug=True, opt=False)
def kernel(d_array):
    device_function(d_array[cuda.threadIdx.x])


class TestDeviceExceptions(CUDATestCase):
    def test_value_error(self):
        d_array = cuda.to_device(np.array([1], dtype=np.int32))
        with self.assertRaises(ValueError):
            kernel[1, 1](d_array)

    def test_type_error(self):
        d_array = cuda.to_device(np.array([2], dtype=np.int32))
        with self.assertRaises(TypeError):
            kernel[1, 1](d_array)

    def test_runtime_error(self):
        d_array = cuda.to_device(np.array([3], dtype=np.int32))
        with self.assertRaises(RuntimeError):
            kernel[1, 1](d_array)

    def test_assertion_error(self):
        d_array = cuda.to_device(np.array([4], dtype=np.int32))
        with self.assertRaises(AssertionError):
            kernel[1, 1](d_array)


if __name__ == '__main__':
    unittest.main()
