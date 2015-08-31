from __future__ import print_function

import contextlib
import io
import os

import numpy

from numba import cuda
from numba import unittest_support as unittest


@contextlib.contextmanager
def redirect_fd(fd):
    """
    Temporarily redirect *fd* to a pipe's write end and return a file object
    wrapping the pipe's read end.
    """
    save = os.dup(fd)
    r, w = os.pipe()
    try:
        os.dup2(w, fd)
        yield io.open(r, "r")
    finally:
        os.close(w)
        os.dup2(save, fd)
        os.close(save)


def cuhello():
    i = cuda.grid(1)
    print(i, 999)


def printfloat():
    i = cuda.grid(1)
    print(i, 23, 34.3, 321)


def cuprintary(A):
    i = cuda.grid(1)
    print("A[", i, "]", A[i])


class TestPrint(unittest.TestCase):

    def test_cuhello(self):
        jcuhello = cuda.jit('void()', debug=False)(cuhello)
        with redirect_fd(1) as stdout:
            jcuhello[2, 3]()
            cuda.synchronize()
        # The output of GPU threads is intermingled, just sanity check it
        out = stdout.read()
        expected = ''.join('%d 999\n' % i for i in range(6))
        self.assertEqual(sorted(out), sorted(expected))

    def test_printfloat(self):
        jprintfloat = cuda.jit('void()', debug=False)(printfloat)
        with redirect_fd(1) as stdout:
            jprintfloat()
            cuda.synchronize()
        self.assertEqual(stdout.read(), "0 23 34.300000 321\n")

    @unittest.skipIf(True, "Print string not implemented yet")
    def test_print_array(self):
        """
        Eyeballing required
        """
        jcuprintary = cuda.jit('void(float32[:])')(cuprintary)
        A = numpy.arange(10, dtype=numpy.float32)
        jcuprintary[2, 5](A)
        cuda.synchronize()


if __name__ == '__main__':
    unittest.main()
