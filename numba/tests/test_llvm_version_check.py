from __future__ import print_function, division, absolute_import
from numba import unittest_support as unittest

class TestLlvmVersion(unittest.TestCase):

    def test_llvm_version(self):
        import llvm

        llvm.__version__ = '0.13.1'
        import numba

        with self.assertRaises(SystemExit):
            llvm.__version__ = '0.9.6'
            reload(numba)
        

if __name__ == '__main__':
    unittest.main()
