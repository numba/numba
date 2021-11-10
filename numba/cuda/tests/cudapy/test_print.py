from numba import cuda
from numba.core.errors import NumbaWarning
from numba.cuda.testing import (captured_cuda_stdout, CUDATestCase,
                                skip_on_cudasim)
import numpy as np
import unittest
import warnings


def cuhello():
    i = cuda.grid(1)
    print(i, 999)
    print(-42)


def printfloat():
    i = cuda.grid(1)
    print(i, 23, 34.75, 321)


def printstring():
    i = cuda.grid(1)
    print(i, "hop!", 999)


def printempty():
    print()


def print_too_many(r):
    print(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10],
          r[11], r[12], r[13], r[14], r[15], r[16], r[17], r[18], r[19], r[20],
          r[21], r[22], r[23], r[24], r[25], r[26], r[27], r[28], r[29], r[30],
          r[31], r[32])


class TestPrint(CUDATestCase):

    def test_cuhello(self):
        jcuhello = cuda.jit('void()', debug=False)(cuhello)
        with captured_cuda_stdout() as stdout:
            jcuhello[2, 3]()
        # The output of GPU threads is intermingled, but each print()
        # call is still atomic
        out = stdout.getvalue()
        lines = sorted(out.splitlines(True))
        expected = ['-42\n'] * 6 + ['%d 999\n' % i for i in range(6)]
        self.assertEqual(lines, expected)

    def test_printfloat(self):
        jprintfloat = cuda.jit('void()', debug=False)(printfloat)
        with captured_cuda_stdout() as stdout:
            jprintfloat[1, 1]()
        # CUDA and the simulator use different formats for float formatting
        self.assertIn(stdout.getvalue(), ["0 23 34.750000 321\n",
                                          "0 23 34.75 321\n"])

    def test_printempty(self):
        cufunc = cuda.jit('void()', debug=False)(printempty)
        with captured_cuda_stdout() as stdout:
            cufunc[1, 1]()
        self.assertEqual(stdout.getvalue(), "\n")

    def test_string(self):
        cufunc = cuda.jit('void()', debug=False)(printstring)
        with captured_cuda_stdout() as stdout:
            cufunc[1, 3]()
        out = stdout.getvalue()
        lines = sorted(out.splitlines(True))
        expected = ['%d hop! 999\n' % i for i in range(3)]
        self.assertEqual(lines, expected)

    @skip_on_cudasim('cudasim can print unlimited output')
    def test_too_many_args(self):
        # Tests that we emit the format string and warn when there are more
        # than 32 arguments, in common with CUDA C/C++ printf - this is due to
        # a limitation in CUDA vprintf, see:
        # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#limitations

        cufunc = cuda.jit(print_too_many)
        r = np.arange(33)
        with captured_cuda_stdout() as stdout:
            with warnings.catch_warnings(record=True) as w:
                cufunc[1, 1](r)

        # Check that the format string was printed instead of formatted garbage
        expected_fmt_string = ' '.join(['%lld' for _ in range(33)])
        self.assertIn(expected_fmt_string, stdout.getvalue())

        # Check for the expected warning about formatting more than 32 items
        for warning in w:
            warnobj = warning.message
            if isinstance(warnobj, NumbaWarning):
                expected = ('CUDA print() cannot print more than 32 items. '
                            'The raw format string will be emitted by the '
                            'kernel instead.')
                if warnobj.msg == expected:
                    break
        else:
            self.fail('Expected a warning for printing more than 32 items')


if __name__ == '__main__':
    unittest.main()
