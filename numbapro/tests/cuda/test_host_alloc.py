import unittest
import numpy as np
from numbapro.cudapipeline import driver

class TestHostAlloc(unittest.TestCase):
    def test_host_alloc_driver(self):
        driver.get_or_create_context()
        mem = driver.HostAllocMemory(1024, map=True)
        buffer = mem.get_host_buffer()
        dtype = np.dtype('int32')
        ary = np.ndarray(shape=1024/dtype.itemsize, dtype=dtype, buffer=buffer)
        ary.fill(1234)
        self.assertTrue(np.all(ary == [1234] * ary.shape[0]))

if __name__ == '__main__':
    unittest.main()
