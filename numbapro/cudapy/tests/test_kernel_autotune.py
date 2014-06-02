from __future__ import print_function, absolute_import
from numbapro.testsupport import unittest
from numbapro import cuda


class TestKernelAutotune(unittest.TestCase):
    def test_kernel_occupancy(self):
        @cuda.jit('void(float32[:])')
        def foo(a):
            i = cuda.grid(1)
            a[i] *= a[i]

        try:
            foo.autotune
        except RuntimeError:
            print('skipped: driver does not support jit info reporting')
        else:
            self.assertTrue(foo[1, 320].occupancy > foo[1, 32].occupancy)


if __name__ == '__main__':
    unittest.main()
