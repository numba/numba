import multiprocessing as mp
import traceback
from numba import njit
from numba.tests.support import (
    TestCase,
    SerialMixin,
    override_config,
    run_in_new_process_caching,
    temp_directory,
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

    def test_omitted(self):

        # Test in a new directory
        cache_dir = temp_directory(self.__class__.__name__)
        ctx = mp.get_context()
        result_queue = ctx.Queue()
        proc = ctx.Process(
            target=omitted_child_test_wrapper,
            args=(result_queue, cache_dir, False),
        )
        proc.start()
        proc.join()
        success, output = result_queue.get()

        # Ensure the child process is completed before checking its output
        if not success:
            self.fail(output)

        self.assertEqual(
            output,
            1000,
            "Omitted function returned an incorrect output"
        )

        proc = ctx.Process(
            target=omitted_child_test_wrapper,
            args=(result_queue, cache_dir, True)
        )
        proc.start()
        proc.join()
        success, output = result_queue.get()

        # Ensure the child process is completed before checking its output
        if not success:
            self.fail(output)

        self.assertEqual(
            output,
            1000,
            "Omitted function returned an incorrect output"
        )


def omitted_child_test_wrapper(result_queue, cache_dir, second_call):
    with override_config("CACHE_DIR", cache_dir):
        @njit(cache=True)
        def test(num=1000):
            return num

        try:
            output = test()
            # If we have a second call, we should have a cache hit.
            # Otherwise, we expect a cache miss.
            if second_call:
                assert test._cache_hits[test.signatures[0]] == 1, \
                    "Cache did not hit as expected"
                assert test._cache_misses[test.signatures[0]] == 0, \
                    "Cache has an unexpected miss"
            else:
                assert test._cache_misses[test.signatures[0]] == 1, \
                    "Cache did not miss as expected"
                assert test._cache_hits[test.signatures[0]] == 0, \
                    "Cache has an unexpected hit"
            success = True
        # Catch anything raised so it can be propagated
        except: # noqa: E722
            output = traceback.format_exc()
            success = False
        result_queue.put((success, output))
