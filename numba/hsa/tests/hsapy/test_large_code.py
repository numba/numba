from __future__ import print_function, absolute_import

import sys
import os
import os.path
import subprocess

import numba
import numba.unittest_support as unittest


class TestLargeCode(unittest.TestCase):

    def test_far_jump(self):
        from . import run_far_branch

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
        except subprocess.CalledProcessError as exc:
            self.assertIn('LLVM ERROR: branch size exceeds simm16',
                          exc.output.decode())
        else:
            self.fail('far branch bug fixed!')
        finally:
            if oldpp is None:
                del os.environ['PYTHONPATH']
            else:
                os.environ['PYTHONPATH'] = oldpp


if __name__ == '__main__':
    unittest.main()
