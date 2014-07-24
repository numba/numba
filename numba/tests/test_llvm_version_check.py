from __future__ import print_function, division, absolute_import
from numba import unittest_support as unittest
import sys

class TestLlvmVersion(unittest.TestCase):

    def test_llvm_version(self):
        import llvm
        import numba
        self.assertTrue(numba.__version__)
        
        if sys.version_info >= (3, 0):
            #git version check
            llvm.__version = '0.12.6-10-g92584ed'
            imp.reload(numba)
            self.assertTrue(numba.__version__)
        
            with self.assertRaises(SystemExit):
                llvm.__version__ = '0.12.5' # llvmpy has to be >= 0.12.6 
                imp.reload(numba)
        else: 
            #git version check
            llvm.__version = '0.12.6-10-g92584ed'
            reload(numba)
            self.assertTrue(numba.__version__)
        
            with self.assertRaises(SystemExit):
                llvm.__version__ = '0.12.5' # llvmpy has to be >= 0.12.6 
                reload(numba)
        

if __name__ == '__main__':
    unittest.main()
