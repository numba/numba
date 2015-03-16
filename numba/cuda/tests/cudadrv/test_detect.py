from __future__ import absolute_import, print_function
from numba import cuda
from numba.cuda.testing import unittest
from numba.tests.support import captured_stdout

class TestCudaDetect(unittest.TestCase):
    def test_cuda_detect(self):
        # exercise the code path
        with captured_stdout() as out:
            cuda.detect()
        output = out.getvalue()
        self.assertIn('Found', output)
        self.assertIn('CUDA devices', output)


if __name__ == '__main__':
    unittest.main()
