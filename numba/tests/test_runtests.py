import sys
import subprocess

from numba import cuda
import unittest

try:
    import git  # from gitpython package
except ImportError:
    has_gitpython = False
else:
    has_gitpython = True


class TestCase(unittest.TestCase):
    """These test cases are meant to test the Numba test infrastructure itself.
    Therefore, the logic used here shouldn't use numba.testing, but only the
    upstream unittest, and run the numba test suite only in a subprocess."""

    def get_testsuite_listing(self, args, *, subp_kwargs=None):
        """
        Use `subp_kwargs` to pass extra argument to `subprocess.check_output`.
        """
        subp_kwargs = subp_kwargs or {}
        cmd = [sys.executable, '-m', 'numba.runtests', '-l'] + list(args)
        out_bytes = subprocess.check_output(cmd, **subp_kwargs)
        lines = out_bytes.decode('UTF-8').splitlines()
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
        self.assertTrue('tests found' in last_line)
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
        self.check_testsuite_size(['numba.tests.test_storeslice'], 2)
        self.check_testsuite_size(['numba.tests.test_nested_calls'], 10)
        # Several modules
        self.check_testsuite_size(['numba.tests.test_nested_calls',
                                   'numba.tests.test_storeslice'], 12)

    def test_subpackage(self):
        self.check_testsuite_size(['numba.tests.npyufunc'], 50)

    def test_random(self):
        self.check_testsuite_size(
            ['--random', '0.1', 'numba.tests.npyufunc'], 5)

    def test_include_exclude_tags(self):
        def get_count(arg_list):
            lines = self.get_testsuite_listing(arg_list)
            self.assertIn('tests found', lines[-1])
            count = int(lines[-1].split()[0])
            self.assertTrue(count > 0)
            return count

        tags = ['long_running', 'long_running, important']

        for tag in tags:
            total = get_count(['numba.tests'])
            included = get_count(['--tags', tag, 'numba.tests'])
            excluded = get_count(['--exclude-tags', tag, 'numba.tests'])
            self.assertEqual(total, included + excluded)

            # check syntax with `=` sign in
            total = get_count(['numba.tests'])
            included = get_count(['--tags=%s' % tag, 'numba.tests'])
            excluded = get_count(['--exclude-tags=%s' % tag, 'numba.tests'])
            self.assertEqual(total, included + excluded)

    def test_check_slice(self):
        tmp = self.get_testsuite_listing(['-j','0,5,1'])
        l = [x for x in tmp if x.startswith('numba.')]
        self.assertEqual(len(l), 5)

    def test_check_slicing_equivalent(self):
        def filter_test(xs):
            return [x for x in xs if x.startswith('numba.')]
        full = filter_test(self.get_testsuite_listing([]))
        sliced = []
        for i in range(3):
            subset = self.get_testsuite_listing(['-j', '{},None,3'.format(i)])
            sliced.extend(filter_test(subset))
        # The tests must be equivalent
        self.assertEqual(sorted(full), sorted(sliced))

    @unittest.skipUnless(has_gitpython, "Requires gitpython")
    def test_gitdiff(self):
        # Check for git
        try:
            subprocess.call("git",
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
        except FileNotFoundError as e:
            self.skipTest("no git available")

        # default
        outs = self.get_testsuite_listing(['-g'])
        self.assertNotIn("Git diff by common ancestor", outs)
        # using ancestor
        outs = self.get_testsuite_listing(['-g=ancestor'])
        self.assertIn("Git diff by common ancestor", outs)
        # misspelled ancestor
        subp_kwargs = dict(stderr=subprocess.DEVNULL)
        with self.assertRaises(subprocess.CalledProcessError):
            self.get_testsuite_listing(['-g=ancest'], subp_kwargs=subp_kwargs)


if __name__ == '__main__':
    unittest.main()
