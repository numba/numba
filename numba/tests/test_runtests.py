#!/usr/bin/env python

import unittest
import subprocess

def check_output(*popenargs, **kwargs):
    # Provide this for backward-compatibility until we drop Python 2.6 support.
    process = subprocess.Popen(stdout=subprocess.PIPE, *popenargs, **kwargs)
    output, unused_err = process.communicate()
    retcode = process.poll()
    if retcode:
        cmd = kwargs.get("args")
        if cmd is None:
            cmd = popenargs[0]
        raise subprocess.CalledProcessError(retcode, cmd)
    return output


class TestCase(unittest.TestCase):
    """These test cases are meant to test the Numba test infrastructure itself.
    Therefore, the logic used here shouldn't use numba.testing, but only the upstream
    unittest, and run the numba test suite only in a subprocess."""
    

    def check_testsuite_size(self, id, minsize, maxsize = None):
        """Check that the reported numbers of tests in 'id' are 
        in the (minsize, maxsize) range, or are equal to minsize if maxsize is None."""

        cmd = ['python', '-m', 'numba.runtests', '-l']
        if id:
            cmd.append(id)
        lines = check_output(cmd).decode().splitlines()
        lines = [line for line in lines if line.strip()]
        last_line = lines[-1]
        self.assertTrue(last_line.endswith('tests found'))
        number = int(last_line.split(' ')[0])
        # There may be some "skipped" messages at the beginning,
        # so do an approximate check.
        self.assertIn(len(lines), range(number + 1, number + 10))
        if maxsize is None:
            self.assertEqual(number, minsize)
        else:
            self.assertGreaterEqual(number, minsize)
            self.assertLessEqual(number, maxsize)

    def test_default(self):
        self.check_testsuite_size('', 6000, 8000)
    def test_all(self):
        self.check_testsuite_size('numba.tests', 6000, 8000)
    def test_cuda(self):
        self.check_testsuite_size('numba.cuda.tests', 0, 400)
    def test_module(self):
        self.check_testsuite_size('numba.tests.test_builtins', 82)

if __name__ == '__main__':
    unittest.main()
    
