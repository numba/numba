import numpy as np
import numba.unittest_support as unittest

from numba.ocl.ocldrv.driver import MemObject
from numba.ocl.ocldrv import cl

class TestOpenCLMemory(unittest.TestCase):
    def setUp(self):
        self.device = cl.default_platform.default_device
        self.context = cl.create_context(self.device.platform, [self.device])

    def tearDown(self):
        del self.context
        del self.device

    def test_simple_buffer_creation(self):
        size = 4096
        buf = self.context.create_buffer(size)
        self.assertIsInstance(buf, MemObject)
        self.assertEqual(buf.size, size)
        self.assertNotEqual(buf.reference_count, 0)
        self.assertEqual(buf.context, self.context) # note: identity will not hold...
        self.assertEqual(buf.host_ptr, None)

    def test_mem_transfer(self):
        #this tests a round trip from host to device and back
        q = self.context.create_command_queue(self.device) # in-order
        size = 16*1024
        test_data = np.random.random_sample(size).astype(np.float32)
        test_result = np.empty_like(test_data)
        buf = self.context.create_buffer(test_data.nbytes)

        q.enqueue_write_buffer(buf, 0, test_data.nbytes, test_data.ctypes.data)
        q.enqueue_read_buffer(buf, 0, test_result.nbytes, test_result.ctypes.data)

        self.assertEqual(np.sum(test_data == test_result), size)

    def test_mem_transfer_2(self):
        #this tests a round trip from host to device and back, including an on-device
        #copy
        q = self.context.create_command_queue(self.device) # in-order
        size = 16*1024
        test_data = np.random.random_sample(size).astype(np.float32)
        test_result = np.empty_like(test_data)
        buf = self.context.create_buffer(test_data.nbytes)
        buf2 = self.context.create_buffer(test_data.nbytes)

        q.enqueue_write_buffer(buf, 0, test_data.nbytes, test_data.ctypes.data)
        q.enqueue_copy_buffer(buf, buf2, 0, 0, test_data.nbytes)
        q.enqueue_read_buffer(buf2, 0, test_result.nbytes, test_result.ctypes.data)

        self.assertEqual(np.sum(test_data == test_result), size)


if __name__ == '__main__':
    unittest.main()