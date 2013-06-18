import unittest
import gc
import numpy as np
from numbapro.cudadrv import driver as cudriver
from numbapro import cuda
import support
cudriver.debug_memory = True

@support.addtest
class TestMemoryLeak(support.CudaTestCase):
    def test_memoryleak(self):
        origalloc = cudriver.debug_memory_alloc
        origfree = cudriver.debug_memory_free

        A = np.arange(100)

        dA = cuda.to_device(A)
        cudriver.print_debug_memory()

        allocated = cudriver.debug_memory_alloc - origalloc
        self.assertTrue(allocated > 0)

        del dA

        cudriver.print_debug_memory()
        freed = cudriver.debug_memory_free - origfree
        print allocated - freed
        self.assertTrue(0 <= allocated - freed <= 1 )

if __name__ == '__main__':
    support.main()
