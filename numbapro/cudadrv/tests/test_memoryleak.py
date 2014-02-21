import numpy as np
from numbapro.cudadrv import old_driver as cudriver
from numbapro import cuda
import support
cudriver.debug_memory = True

@support.addtest
class TestMemoryLeak(support.CudaTestCase):
    def test_memoryleak(self):
        cudriver.flush_pending_free()
        origalloc = cudriver.debug_memory_alloc
        origfree = cudriver.debug_memory_free

        A = np.arange(100)

        dA = cuda.to_device(A)
        cudriver.print_debug_memory()

        allocated = cudriver.debug_memory_alloc - origalloc
        self.assertTrue(allocated > 0)

        del dA
        cudriver.flush_pending_free()


        cudriver.print_debug_memory()
        freed = cudriver.debug_memory_free - origfree
        print allocated - freed
        self.assertTrue(0 <= allocated - freed <= 1 )

if __name__ == '__main__':
    support.main()
