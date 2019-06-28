from __future__ import print_function, absolute_import, division

import sys
import multiprocessing as mp

from numba import njit
from .support import (
    unittest,
    TestCase,
    SerialMixin,
    run_in_new_process_caching
)


_py34_or_later = sys.version_info[:2] >= (3, 4)
_has_mp_get_context = hasattr(mp, 'get_context')
_skip_no_unicode = unittest.skipUnless(
    _py34_or_later,
    "unicode requires py3.4+",
)
_skip_no_mp_spawn = unittest.skipUnless(
    _has_mp_get_context,
    "requires multiprocessing.get_context",
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


class TestCaching(SerialMixin, TestCase):
    def run_test(self, func):
        func()
        res = run_in_new_process_caching(func)
        self.assertEqual(res['exitcode'], 0)

    @_skip_no_unicode
    @_skip_no_mp_spawn
    def test_constant_unicode_cache(self):
        self.run_test(check_constant_unicode_cache)

    @_skip_no_unicode
    @_skip_no_mp_spawn
    def test_dict_cache(self):
        self.run_test(check_dict_cache)
