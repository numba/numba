from __future__ import print_function, division, absolute_import

import imp
import sys

from numba import unittest_support as unittest


class TestLlvmVersion(unittest.TestCase):

    def test_llvm_version(self):
        # test the system its running on
        import llvm
        import numba
        self.assertTrue(numba.__version__)

        llvm_version = llvm.__version__
        def cleanup():
            llvm.__version__ = llvm_version
        self.addCleanup(cleanup)

        # explicitly test all 4 cases of version string
        version_pass = '0.12.6'
        git_version_pass = '0.12.6-10-g92584ed'
        rc_version_pass = '0.12.7rc1'
        version_fail = '0.12.5'
        git_version_fail = '0.12.5-10-g92584ed'

        ver_pass = (version_pass, git_version_pass, rc_version_pass)
        ver_fail = (version_fail, git_version_fail)
        for v in ver_pass:
            llvm.__version__ = v
            imp.reload(numba)
            self.assertTrue(numba.__version__)

        for v in ver_fail:
            with self.assertRaises(ImportError):
                llvm.__version__ = v
                imp.reload(numba)


if __name__ == '__main__':
    unittest.main()
