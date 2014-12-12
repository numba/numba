from __future__ import print_function, division, absolute_import

import imp
import sys

from numba import unittest_support as unittest


class TestLlvmVersion(unittest.TestCase):

    def test_llvmlite_version(self):
        # test the system it's running on
        import llvmlite
        import numba
        self.assertTrue(numba.__version__)

        llvmlite_version = llvmlite.__version__
        def cleanup():
            llvmlite.__version__ = llvmlite_version
        self.addCleanup(cleanup)

        # explicitly test all 4 cases of version string
        version_pass = '0.1.0'
        git_version_pass = '0.1.0-10-g92584ed'
        rc_version_pass = '0.1.1rc1'
        version_fail = '0.0.9'
        git_version_fail = '0.0.9-10-g92584ed'

        ver_pass = (version_pass, git_version_pass, rc_version_pass)
        ver_fail = (version_fail, git_version_fail)
        for v in ver_pass:
            llvmlite.__version__ = v
            imp.reload(numba)
            self.assertTrue(numba.__version__)

        for v in ver_fail:
            with self.assertRaises(ImportError):
                llvmlite.__version__ = v
                imp.reload(numba)


if __name__ == '__main__':
    unittest.main()
