from numba import njit
from numba.tests.support import (
    TestCase,
    SerialMixin,
    run_in_new_process_caching
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

    def test_constant_unicode_cache(self):
        self.run_test(check_constant_unicode_cache)

    def test_dict_cache(self):
        self.run_test(check_dict_cache)
