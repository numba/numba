import unittest
from numbapro.cudapipeline.ndarray import SmallMemoryManager

class TestManagedMemory(unittest.TestCase):
    def test_smallmemorymanager(self):
        smm = SmallMemoryManager(nbytes=256)
        val0 = 1, 2, 3
        mm0 = smm.obtain(val0)
        mm1 = smm.obtain(val0)
        #print smm.usemap
        val1 = 3, 2, 1
        mm2 = smm.obtain(val1)
        print smm.usemap
        del mm0
        del mm1
        del mm2

        mm3 = smm.obtain(val0)
        print smm.usemap

        del mm3

        print '==loop=='
        for val in range(10):
            print smm.usemap
            mm = smm.obtain((val,))

if __name__ == '__main__':
    unittest.main()
