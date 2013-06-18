import unittest

class CudaTestCase(unittest.TestCase):
    '''Safe guard all CUDA testcase from memory leak problem.
    '''

    def setUp(self):
        from numbapro.cudadrv import driver
        driver.get_or_create_context()
        driver.debug_memory = True
        self._start_alloc = driver.debug_memory_alloc
        self._start_free = driver.debug_memory_free

    def tearDown(self):
        from numbapro.cudadrv import driver
        driver.debug_memory = True
        alloced = driver.debug_memory_alloc - self._start_alloc
        freed = driver.debug_memory_free - self._start_free
        self.assertTrue(alloced - freed <= 1,  # 1 alloc difference for SMM
                         "Memory leak detected\n"
                         "alloced: %d freed: %d" % (alloced, freed))


