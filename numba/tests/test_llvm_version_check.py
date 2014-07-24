from __future__ import print_function, division, absolute_import
from numba import unittest_support as unittest

class TestLlvmVersion(unittest.TestCase):

    def test_llvm_version(self):
        import llvm
        import numba
        self.assertTrue(numba.__version__)

        with self.assertRaises(SystemExit):
            llvm.__version__ = '0.12.5' # llvm has to be >= 12.6 
            reload(numba)
        

if __name__ == '__main__':
    unittest.main()
