"""
Test gdb
"""
from __future__ import print_function
import ctypes
import os
import subprocess
import sys
import threading

from numba import njit, gdb, gdb_init, gdb_breakpoint, prange, testing

from .support import (TestCase, override_env_config, captured_stdout, tag)

class TestGdbBindImpls(TestCase):

    def test_gdb_cmd_lang_impl(self):
        @njit(debug=True)
        def impl(a):
            gdb('-ex', 'set confirm off', '-ex', 'c', '-ex', 'q')
            b = a + 1
            c = a * 2.34
            d = (a, b, c)
            print(a, b, c, d)
        impl(10)

    def test_gdb_split_init_and_break_impl(self):
        @njit(debug=True)
        def impl(a):
            gdb_init('-ex', 'set confirm off', '-ex', 'c', '-ex', 'q')
            b = a + 1
            c = a * 2.34
            d = (a, b, c)
            gdb_breakpoint()
            print(a, b, c, d)
        impl(10)

    def test_gdb_split_init_and_break_impl_w_parallel(self):
        @njit(debug=True, parallel=True)
        def impl(a):
            gdb_init('-ex', 'set confirm off', '-ex', 'c', '-ex', 'q')
            a += 3
            for i in prange(4):
                b = a + 1
                c = a * 2.34
                d = (a, b, c)
                gdb_breakpoint()
                print(a, b, c, d)
        impl(10)

class TestGdbBinding(TestCase):

    # test mutates env
    _numba_parallel_test_ = False

    _DEBUG = True

    def run_cmd(self, cmdline, env, kill_is_ok=False):
        popen = subprocess.Popen(cmdline,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 env=env,
                                 shell=True)
        # finish in 10s or kill it, there's no work being done
        def kill():
            popen.stdout.flush()
            popen.stderr.flush()
            popen.kill()
        timeout = threading.Timer(20., kill)
        try:
            timeout.start()
            out, err = popen.communicate()
            retcode = popen.returncode
            if retcode != 0:
                raise AssertionError(
                "process failed with code %s: stderr follows\n%s\n" %
                (retcode, err.decode()))
            return out.decode(), err.decode()
        finally:
            timeout.cancel()
        return None, None

    def run_test_in_separate_process(self, test, **kwargs):
        env_copy = os.environ.copy()
        env_copy['NUMBA_OPT'] = '1'
        cmdline = [sys.executable, "-m", "numba.runtests", test]
        return self.run_cmd(' '.join(cmdline), env_copy, **kwargs)

    @classmethod
    def _inject(cls, name):
        themod = TestGdbBindImpls.__module__
        thecls = TestGdbBindImpls.__name__
        # strip impl
        methname = name.replace('_impl','')
        injected_method = '%s.%s.%s' % (themod, thecls, name)

        def test_template(self):
            o, e = self.run_test_in_separate_process(injected_method)
            self.assertIn('GNU gdb', o)
            self.assertIn('OK', e)
            self.assertTrue('FAIL' not in e)
            self.assertTrue('ERROR' not in e)
        setattr(cls, methname, test_template)


    @classmethod
    def generate(cls):
        for name in dir(TestGdbBindImpls):
            if name.startswith('test_gdb'):
                cls._inject(name)

TestGdbBinding.generate()

if __name__ == '__main__':
    unittest.main()

