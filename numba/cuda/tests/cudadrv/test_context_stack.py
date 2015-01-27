from __future__ import print_function
from numba import cuda
from numba.cuda.testing import unittest


class TestContextStack(unittest.TestCase):
    def setUp(self):
        # Reset before testing
        cuda.close()

    def test_gpus_current(self):
        self.assertIs(cuda.gpus.current, None)
        with cuda.gpus[0]:
            self.assertEqual(cuda.gpus.current.id, 0)

    def test_gpus_len(self):
        self.assertGreater(len(cuda.gpus), 0)

    def test_gpus_iter(self):
        gpulist = list(cuda.gpus)
        self.assertGreater(len(gpulist), 0)


if __name__ == '__main__':
    unittest.main()
