from __future__ import print_function

import numbers

from numba import cuda
from numba.cuda.testing import unittest, SerialMixin


class TestContextStack(SerialMixin, unittest.TestCase):
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


class TestContextAPI(SerialMixin, unittest.TestCase):

    def test_context_memory(self):
        mem = cuda.current_context().get_memory_info()

        self.assertIsInstance(mem.free, numbers.Number)
        self.assertEquals(mem.free, mem[0])

        self.assertIsInstance(mem.total, numbers.Number)
        self.assertEquals(mem.total, mem[1])

        self.assertLessEqual(mem.free, mem.total)


if __name__ == '__main__':
    unittest.main()
