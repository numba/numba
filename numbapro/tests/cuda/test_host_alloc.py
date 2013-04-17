import unittest
import numpy as np
from numbapro.cudapipeline import driver

class TestHostAlloc(unittest.TestCase):
    def setUp(self):
        driver.get_or_create_context()

    def test_host_alloc_driver(self):
        n = 32
        mem = driver.HostAllocMemory(n, mapped=True)

        buf = memoryview(mem)

        dtype = np.dtype(np.uint8)
        ary = np.ndarray(shape=n / dtype.itemsize, dtype=dtype, buffer=mem)

        magic = 0xab
        driver.device_memset(mem, magic, n)

        self.assertTrue(np.all(ary == magic))

        ary.fill(n)

        recv = np.empty_like(ary)

        driver.device_to_host(recv, mem, ary.size)

        self.assertTrue(np.all(ary == recv))
        self.assertTrue(np.all(recv == n))

if __name__ == '__main__':
    unittest.main()
