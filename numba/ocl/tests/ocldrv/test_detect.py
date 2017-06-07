from __future__ import absolute_import, print_function
from numba import ocl
from numba.ocl.testing import unittest
from numba.tests.support import captured_stdout

class TestOclDetect(unittest.TestCase):
    def test_ocl_detect(self):
        # exercise the code path
        with captured_stdout() as out:
            ocl.detect()
        output = out.getvalue()
        self.assertIn('Found', output)
        self.assertIn('OCL devices', output)


if __name__ == '__main__':
    unittest.main()

