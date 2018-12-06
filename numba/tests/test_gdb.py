"""
Tests gdb bindings
"""
from __future__ import print_function
import ctypes
import os
import subprocess
import sys
import threading
from itertools import permutations

from numba import njit, gdb, gdb_init, gdb_breakpoint, prange, config, errors
from numba import unittest_support as unittest

from .support import (TestCase, override_env_config, captured_stdout, tag)
from .test_parfors import skip_unsupported as parfors_skip_unsupported
from .test_parfors import linux_only

_linux = sys.platform.startswith('linux')

_gdb_cond = os.environ.get('GDB_TEST', None) == '1'
needs_gdb_harness = unittest.skipUnless(_gdb_cond, "needs gdb harness")
_gdbloc = config.GDB_BINARY
_has_gdb = (os.path.exists(_gdbloc) and os.path.isfile(_gdbloc))
needs_gdb = unittest.skipUnless(_has_gdb, "gdb binary is required")


@linux_only
class TestGdbBindImpls(TestCase):
    """
    Contains unit test implementations for gdb binding testing. Test must be
    decorated with `@needs_gdb_harness` to prevent their running under normal
    test conditions, the test methods must also end with `_impl` to be
    considered for execution. The tests themselves are invoked by the
    `TestGdbBinding` test class through the parsing of this class for test
    methods and then running the discovered tests in a separate process.
    """

    @needs_gdb_harness
    def test_gdb_cmd_lang_impl(self):
        @njit(debug=True)
        def impl(a):
            gdb('-ex', 'set confirm off', '-ex', 'c', '-ex', 'q')
            b = a + 1
            c = a * 2.34
            d = (a, b, c)
            print(a, b, c, d)
        impl(10)

    @needs_gdb_harness
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

    @parfors_skip_unsupported
    @needs_gdb_harness
    def test_gdb_split_init_and_break_w_parallel_impl(self):
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

@linux_only
@needs_gdb
class TestGdbBinding(TestCase):
    """
    This test class is used to generate tests which will run the test cases
    defined in TestGdbBindImpls in isolated subprocesses, this is for safety
    in case something goes awry.
    """

    # test mutates env
    _numba_parallel_test_ = False

    _DEBUG = True

    def run_cmd(self, cmdline, env, kill_is_ok=False):
        popen = subprocess.Popen(cmdline,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 env=env,
                                 shell=True)
        # finish in 20s or kill it, there's no work being done
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
        # Set GDB_TEST to permit the execution of tests decorated with
        # @needs_gdb_harness
        env_copy['GDB_TEST'] = '1'
        cmdline = [sys.executable, "-m", "numba.runtests", test]
        return self.run_cmd(' '.join(cmdline), env_copy, **kwargs)

    @classmethod
    def _inject(cls, name):
        themod = TestGdbBindImpls.__module__
        thecls = TestGdbBindImpls.__name__
        # strip impl
        assert name.endswith('_impl')
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

@linux_only
@needs_gdb
class TestGdbMisc(TestCase):

    def test_call_gdb_twice(self):
        def gen(f1, f2):
            @njit
            def impl():
                a = 1
                f1()
                b = 2
                f2()
                return a + b
            return impl

        msg_head = "Calling either numba.gdb() or numba.gdb_init() more than"

        def check(func):
            with self.assertRaises(errors.UnsupportedError) as raises:
                func()
            self.assertIn(msg_head, str(raises.exception))

        for g1, g2 in permutations([gdb, gdb_init]):
            func  = gen(g1, g2)
            check(func)

        @njit
        def use_globals():
            a = 1
            gdb()
            b = 2
            gdb_init()
            return a + b

        check(use_globals)


if __name__ == '__main__':
    unittest.main()

