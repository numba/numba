import unittest
from numbapro import testsupport
testsupport.set_base(globals())

class CudaTestCase(unittest.TestCase):
    '''Safe guard all CUDA testcase from memory leak problem.
    '''

    def setUp(self):
        from numbapro.cudadrv import old_driver
        old_driver.get_or_create_context()
        old_driver.debug_memory = True
        self._start_alloc = old_driver.debug_memory_alloc
        self._start_free = old_driver.debug_memory_free

    def tearDown(self):
        from numbapro.cudadrv import old_driver
        old_driver.flush_pending_free()
        old_driver.debug_memory = True
        alloced = old_driver.debug_memory_alloc - self._start_alloc
        freed = old_driver.debug_memory_free - self._start_free
        self.assertTrue(alloced - freed <= 1,  # 1 alloc difference for SMM
                         "Memory leak detected\n"
                         "alloced: %d freed: %d" % (alloced, freed))


