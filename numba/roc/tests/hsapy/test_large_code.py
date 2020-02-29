import sys
import os
import os.path
import subprocess
import math

import numba
import unittest

class TestLargeCode(unittest.TestCase):

    def test_far_jump(self):
        from numba.roc.tests.hsapy import run_far_branch

        pyinterp = sys.executable
        numba_dir = os.path.abspath(os.path.join(os.path.dirname(numba.__file__), os.pardir))
        script, ext = os.path.splitext(os.path.relpath(run_far_branch.__file__, numba_dir))
        script = script.replace(os.path.sep, '.')
        args = [pyinterp, script]
        cmd = '{} -m {}'.format(*args)

        oldpp = os.environ.get('PYTHONPATH')
        os.environ['PYTHONPATH'] = numba_dir

        try:
            subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
        finally:
            if oldpp is None:
                del os.environ['PYTHONPATH']
            else:
                os.environ['PYTHONPATH'] = oldpp

if __name__ == '__main__':
    unittest.main()
