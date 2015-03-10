
from __future__ import print_function, division, absolute_import

import numpy as np
from numba import unittest_support as unittest
from numba import mviewbuf

class TestMviewBuf(unittest.TestCase):
    def test_buffer_proxy(self):
        dt = np.dtype([('a', np.int32), ('b', np.float32)])
        rec = np.recarray(1, dt)[0]
        bp = mviewbuf.BufferProxy(rec)
        ptr = mviewbuf.memoryview_get_buffer(bp)

        # Check that the returned buffer is a nonzero integer
        self.assertIsInstance(ptr, int)
        self.assertNotEqual(0, ptr)

        # Check that getting further buffers returns the same pointer
        ptr2 = mviewbuf.memoryview_get_buffer(bp)
        self.assertEqual(ptr, ptr2)

if __name__ == '__main__':
    unittest.main()
