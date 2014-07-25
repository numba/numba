from __future__ import print_function, division, absolute_import
from numba import unittest_support as unittest
import sys

class TestLlvmVersion(unittest.TestCase):
    def test_llvm_version(self):
        # test the system its running on
        import llvm
        import numba
        self.assertTrue(numba.__version__)
     
        if sys.version_info >= (3, 4):
            from importlib import reload
        elif (sys.version_info[0], sys.version_info[1]) == (3, 3):
            from imp import reload
        else:
            from __builtin__ import reload 

        # explicitly test all 4 cases of version string
        version_pass = '0.12.6'
        git_version_pass = '0.12.6-10-g92584ed'
        version_fail = '0.12.5'
        git_version_fail = '0.12.5-10-g92584ed'
        ver_pass = (version_pass, git_version_pass)
        ver_fail = (version_fail, git_version_fail)
        for v in ver_pass:
            llvm.__version = v
            reload(numba)
            self.assertTrue(numba.__version__)
        for v in ver_fail: 
            with self.assertRaises(SystemExit):
                llvm.__version__ = v
                reload(numba)

if __name__ == '__main__':
    unittest.main()
