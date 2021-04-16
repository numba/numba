from numba import cuda
from numba.core.extending import overload
from numba.cuda.testing import (CUDATestCase, captured_cuda_stdout,
                                skip_on_cudasim, unittest)


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


@overload(generic_func_1, hardware='generic')
def ol_generic_func_1():
    def impl():
        print("Generic function 1")
    return impl


@overload(cuda_func_1, hardware='cuda')
def ol_cuda_func_1():
    def impl():
        print("CUDA function 1")
    return impl


@overload(generic_func_2, hardware='generic')
def ol_generic_func_2():
    def impl():
        print("Generic function 2")
    return impl


@overload(cuda_func_2, hardware='cuda')
def ol_cuda_func():
    def impl():
        print("CUDA function 2")
    return impl


@overload(generic_calls_generic, hardware='generic')
def ol_generic_calls_generic():
    def impl():
        print("Generic calls generic")
        generic_func_1()
    return impl


@overload(generic_calls_cuda, hardware='generic')
def ol_generic_calls_cuda():
    def impl():
        print("Generic calls CUDA")
        cuda_func_1()
    return impl


@overload(cuda_calls_generic, hardware='cuda')
def ol_cuda_calls_generic():
    def impl():
        print("CUDA calls generic")
        generic_func_1()
    return impl


@overload(cuda_calls_cuda, hardware='cuda')
def ol_cuda_calls_cuda():
    def impl():
        print("CUDA calls CUDA")
        cuda_func_1()
    return impl


@overload(hardware_overloaded, hardware='generic')
def ol_hardware_overloaded_generic():
    def impl():
        print("Generic hardware overloaded function")
    return impl


@overload(hardware_overloaded, hardware='cuda')
def ol_hardware_overloaded_cuda():
    def impl():
        print("CUDA hardware overloaded function")
    return impl


@overload(generic_calls_hardware_overloaded, hardware='generic')
def ol_generic_calls_hardware_overloaded():
    def impl():
        print("Generic calls hardware overloaded")
        hardware_overloaded()
    return impl


@overload(cuda_calls_hardware_overloaded, hardware='cuda')
def ol_cuda_calls_hardware_overloaded():
    def impl():
        print("CUDA calls hardware overloaded")
        hardware_overloaded()
    return impl


@overload(hardware_overloaded_calls_hardware_overloaded, hardware='generic')
def ol_generic_calls_hardware_overloaded_generic():
    def impl():
        print("Generic hardware overloaded calls hardware overloaded")
        hardware_overloaded()
    return impl


@overload(hardware_overloaded_calls_hardware_overloaded, hardware='cuda')
def ol_generic_calls_hardware_overloaded_cuda():
    def impl():
        print("CUDA hardware overloaded calls hardware overloaded")
        hardware_overloaded()
    return impl


@skip_on_cudasim('Overloading not supported in cudasim')
class TestOverload(CUDATestCase):
    def test_generic(self):
        @cuda.jit
        def call_kernel():
            generic_func_1()

        with captured_cuda_stdout() as stdout:
            call_kernel[1, 1]()
            cuda.synchronize()

        out = stdout.getvalue()
        self.assertIn('Generic function 1', out)
        self.assertNotIn('Generic function 2', out)
        self.assertNotIn('CUDA', out)

    def test_cuda(self):
        @cuda.jit
        def call_kernel():
            cuda_func_1()

        with captured_cuda_stdout() as stdout:
            call_kernel[1, 1]()
            cuda.synchronize()

        out = stdout.getvalue()
        self.assertIn('CUDA function 1', out)
        self.assertNotIn('CUDA function 2', out)
        self.assertNotIn('Generic', out)

    def test_generic_and_cuda(self):
        @cuda.jit
        def call_kernel():
            generic_func_1()
            cuda_func_1()

        with captured_cuda_stdout() as stdout:
            call_kernel[1, 1]()
            cuda.synchronize()

        out = stdout.getvalue()
        self.assertIn('Generic function 1', out)
        self.assertIn('CUDA function 1', out)
        self.assertNotIn('2', out)

    def test_call_two_generic_calls(self):
        @cuda.jit
        def call_kernel():
            generic_func_1()
            generic_func_2()

        with captured_cuda_stdout() as stdout:
            call_kernel[1, 1]()
            cuda.synchronize()

        out = stdout.getvalue()
        self.assertIn('Generic function 1', out)
        self.assertIn('Generic function 2', out)

    def test_call_two_cuda_calls(self):
        @cuda.jit
        def call_kernel():
            cuda_func_1()
            cuda_func_2()

        with captured_cuda_stdout() as stdout:
            call_kernel[1, 1]()
            cuda.synchronize()

        out = stdout.getvalue()
        self.assertIn('CUDA function 1', out)
        self.assertIn('CUDA function 2', out)

    def test_generic_calls_generic(self):
        @cuda.jit
        def call_kernel():
            generic_calls_generic()

        with captured_cuda_stdout() as stdout:
            call_kernel[1, 1]()
            cuda.synchronize()

        out = stdout.getvalue()
        self.assertIn("Generic calls generic", out)
        self.assertIn("Generic function 1", out)
        self.assertNotIn("CUDA", out)

    def test_generic_calls_cuda(self):
        @cuda.jit
        def call_kernel():
            generic_calls_cuda()

        with captured_cuda_stdout() as stdout:
            call_kernel[1, 1]()
            cuda.synchronize()

        out = stdout.getvalue()
        self.assertIn("Generic calls CUDA", out)
        self.assertIn("CUDA function 1", out)

    def test_cuda_calls_generic(self):
        @cuda.jit
        def call_kernel():
            cuda_calls_generic()

        with captured_cuda_stdout() as stdout:
            call_kernel[1, 1]()
            cuda.synchronize()

        out = stdout.getvalue()
        self.assertIn("CUDA calls generic", out)
        self.assertIn("Generic function 1", out)

    def test_cuda_calls_cuda(self):
        @cuda.jit
        def call_kernel():
            cuda_calls_cuda()

        with captured_cuda_stdout() as stdout:
            call_kernel[1, 1]()
            cuda.synchronize()

        out = stdout.getvalue()
        self.assertIn("CUDA calls CUDA", out)
        self.assertIn("CUDA function 1", out)
        self.assertNotIn("Generic", out)

    def test_call_hardware_overloaded(self):
        @cuda.jit
        def call_kernel():
            hardware_overloaded()

        with captured_cuda_stdout() as stdout:
            call_kernel[1, 1]()
            cuda.synchronize()

        out = stdout.getvalue()
        self.assertIn("CUDA hardware overloaded function", out)
        self.assertNotIn("Generic hardware overloaded function", out)

    def test_generic_calls_hardware_overloaded(self):
        @cuda.jit
        def call_kernel():
            generic_calls_hardware_overloaded()

        with captured_cuda_stdout() as stdout:
            call_kernel[1, 1]()
            cuda.synchronize()

        out = stdout.getvalue()
        self.assertIn("Generic calls hardware overloaded", out)
        self.assertIn("CUDA hardware overloaded function", out)
        self.assertNotIn("Generic hardware overloaded function", out)

    def test_cuda_calls_hardware_overloaded(self):
        @cuda.jit
        def call_kernel():
            cuda_calls_hardware_overloaded()

        with captured_cuda_stdout() as stdout:
            call_kernel[1, 1]()
            cuda.synchronize()

        out = stdout.getvalue()
        self.assertIn("CUDA calls hardware overloaded", out)
        self.assertIn("CUDA hardware overloaded function", out)
        self.assertNotIn("Generic hardware overloaded function", out)

    def test_hardware_overloaded_calls_hardware_overloaded(self):
        @cuda.jit
        def call_kernel():
            hardware_overloaded_calls_hardware_overloaded()

        with captured_cuda_stdout() as stdout:
            call_kernel[1, 1]()
            cuda.synchronize()

        out = stdout.getvalue()
        self.assertIn("CUDA hardware overloaded calls hardware overloaded", out)
        self.assertIn("CUDA hardware overloaded function", out)
        self.assertNotIn("Generic", out)


if __name__ == '__main__':
    unittest.main()
