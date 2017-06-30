from __future__ import print_function
from numba import ocl
from numba.ocl.testing import unittest


class TestContextStack(unittest.TestCase):
    def setUp(self):
        # Reset before testing
        ocl.close()

    def test_gpus_current(self):
        self.assertIs(ocl.gpus.current, None)
        with ocl.gpus[0]:
            self.assertGreater(ocl.gpus.current.id, 0)

    def test_gpus_len(self):
        self.assertGreater(len(ocl.gpus), 0)

    def test_gpus_iter(self):
        gpulist = list(ocl.gpus)
        self.assertGreater(len(gpulist), 0)


if __name__ == '__main__':
    unittest.main()
