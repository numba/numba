from __future__ import print_function, absolute_import, division

import sys
import os
import multiprocessing as mp
import traceback

from numba import njit
from .support import (
    TestCase,
    temp_directory,
    override_env_config,
    captured_stdout,
    captured_stderr,
)


def constant_unicode_cache():
    c = "abcd"
    return hash(c), c


def check_constant_unicode_cache():
    pyfunc = constant_unicode_cache
    cfunc = njit(cache=True)(pyfunc)
    exp_hv, exp_str = pyfunc()
    got_hv, got_str = cfunc()
    assert exp_hv == got_hv
    assert exp_str == got_str


def dict_cache():
    return {'a': 1, 'b': 2}


def check_dict_cache():
    pyfunc = dict_cache
    cfunc = njit(cache=True)(pyfunc)
    exp = pyfunc()
    got = cfunc()
    assert exp == got


class TestCaching(TestCase):
    def run_test(self, func):
        func()
        ctx = mp.get_context('spawn')
        qout = ctx.Queue()
        cache_dir = temp_directory(__name__)
        with override_env_config('NUMBA_CACHE_DIR', cache_dir):
            proc = ctx.Process(target=_remote_runner, args=[func, qout])
            proc.start()
            stdout = qout.get()
            stderr = qout.get()
            if stdout.strip():
                print()
                print('STDOUT'.center(80, '-'))
                print(stdout)
            if stderr.strip():
                print()
                print('STDERR'.center(80, '-'))
                print(stderr)
            proc.join()
            self.assertEqual(proc.exitcode, 0)


    # The following is used to auto populate test methods into this class

    def _make_test(fn):
        def udt(self):
            self.run_test(fn)
        return udt

    for k, v in globals().items():
        prefix = 'check_'
        if k.startswith(prefix):
            locals()['test_' + k[len(prefix):]] = _make_test(v)


def _remote_runner(fn, qout):
    with captured_stderr() as stderr:
        with captured_stdout() as stdout:
            try:
                fn()
            except Exception:
                print(traceback.format_exc(), file=sys.stderr)
                exitcode = 1
            else:
                exitcode = 0
        qout.put(stdout.getvalue())
    qout.put(stderr.getvalue())
    sys.exit(exitcode)


def _remote_wrapper(fn):
    _remote_wrapper()

