from __future__ import division, print_function

import sys
import subprocess

from numba import unittest_support as unittest
from numba import cuda


class TestCase(unittest.TestCase):
    """These test cases are meant to test the Numba test infrastructure itself.
    Therefore, the logic used here shouldn't use numba.testing, but only the
    upstream unittest, and run the numba test suite only in a subprocess."""

    def get_testsuite_listing(self, args):
        cmd = ['python', '-m', 'numba.runtests', '-l'] + list(args)
        lines = subprocess.check_output(cmd).decode('UTF-8').splitlines()
        lines = [line for line in lines if line.strip()]
        return lines

    def check_listing_prefix(self, prefix):
        listing = self.get_testsuite_listing([prefix])
        for ln in listing[:-1]:
            errmsg = '{!r} not startswith {!r}'.format(ln, prefix)
            self.assertTrue(ln.startswith(prefix), msg=errmsg)

    def check_testsuite_size(self, args, minsize):
        """
        Check that the reported numbers of tests are at least *minsize*.
        """
        lines = self.get_testsuite_listing(args)
        last_line = lines[-1]
        self.assertTrue(last_line.endswith('tests found'))
        number = int(last_line.split(' ')[0])
        # There may be some "skipped" messages at the beginning,
        # so do an approximate check.
        self.assertIn(len(lines), range(number + 1, number + 10))
        self.assertGreaterEqual(number, minsize)
        return lines

    def check_all(self, ids):
        lines = self.check_testsuite_size(ids, 5000)
        # CUDA should be included by default
        self.assertTrue(any('numba.cuda.tests.' in line for line in lines))
        # As well as subpackage
        self.assertTrue(
            any('numba.tests.npyufunc.test_' in line for line in lines),
            )

    def test_default(self):
        self.check_all([])

    def test_all(self):
        self.check_all(['numba.tests'])

    def test_cuda(self):
        # Even without CUDA enabled, there is at least one test
        # (in numba.cuda.tests.nocuda)
        minsize = 100 if cuda.is_available() else 1
        self.check_testsuite_size(['numba.cuda.tests'], minsize)

    @unittest.skipIf(not cuda.is_available(), "NO CUDA")
    def test_cuda_submodules(self):
        self.check_listing_prefix('numba.cuda.tests.cudadrv')
        self.check_listing_prefix('numba.cuda.tests.cudapy')
        self.check_listing_prefix('numba.cuda.tests.nocuda')
        self.check_listing_prefix('numba.cuda.tests.cudasim')

    def test_module(self):
        self.check_testsuite_size(['numba.tests.test_utils'], 3)
        self.check_testsuite_size(['numba.tests.test_nested_calls'], 5)
        # Several modules
        self.check_testsuite_size(['numba.tests.test_nested_calls',
                                   'numba.tests.test_utils'], 13)

    def test_subpackage(self):
        self.check_testsuite_size(['numba.tests.npyufunc'], 50)

    @unittest.skipIf(sys.version_info < (3, 4),
                     "'--random' only supported on Python 3.4 or higher")
    def test_random(self):
        self.check_testsuite_size(
            ['--random', '0.1', 'numba.tests.npyufunc'], 5)

    @unittest.skipIf(sys.version_info < (3, 4),
                     "'--tags' only supported on Python 3.4 or higher")
    def test_tags(self):
        self.check_testsuite_size(
            ['--tags', 'important', 'numba.tests.npyufunc'], 20,
            )


if __name__ == '__main__':
    unittest.main()
