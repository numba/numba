from __future__ import print_function, absolute_import, division

import os.path
import re
from contextlib import contextmanager

import numpy as np

from numba import unittest_support as unittest
from numba import config

from ..support import MemoryLeakMixin, captured_stdout
from ..test_dispatcher import BaseCacheTest


class UfuncCacheTest(BaseCacheTest):
    """
    Since the cache stats is not exposed by ufunc, we test by looking at the
    cache debug log.
    """
    here = os.path.dirname(__file__)
    usecases_file = os.path.join(here, "cache_usecases.py")
    modname = "ufunc_caching_test_fodder"

    regex_data_saved = re.compile(r'\[cache\] data saved to')
    regex_index_saved = re.compile(r'\[cache\] index saved to')

    regex_data_loaded = re.compile(r'\[cache\] data loaded from')
    regex_index_loaded = re.compile(r'\[cache\] index loaded from')

    @contextmanager
    def capture_cache_log(self):
        with captured_stdout() as out:
            old, config.DEBUG_CACHE = config.DEBUG_CACHE, True
            yield out
            config.DEBUG_CACHE = old

    def check_cache_saved(self, cachelog, count):
        """
        Check number of cache-save were issued
        """
        data_saved = self.regex_data_saved.findall(cachelog)
        index_saved = self.regex_index_saved.findall(cachelog)
        self.assertEqual(len(data_saved), count)
        self.assertEqual(len(index_saved), count)

    def check_cache_loaded(self, cachelog, count):
        """
        Check number of cache-load were issued
        """
        data_loaded = self.regex_data_loaded.findall(cachelog)
        index_loaded = self.regex_index_loaded.findall(cachelog)
        self.assertEqual(len(data_loaded), count)
        self.assertEqual(len(index_loaded), count)

    def check_ufunc_cache(self, usecase_name, n_overloads, **kwargs):
        """
        Check number of cache load/save.
        There should be one per overloaded version.
        """
        mod = self.import_module()
        usecase = getattr(mod, usecase_name)
        # New cache entry saved
        with self.capture_cache_log() as out:
            new_ufunc = usecase(**kwargs)
        cachelog = out.getvalue()
        self.check_cache_saved(cachelog, count=n_overloads)

        # Use cached version
        with self.capture_cache_log() as out:
            cached_ufunc = usecase(**kwargs)
        cachelog = out.getvalue()
        self.check_cache_loaded(cachelog, count=n_overloads)

        return new_ufunc, cached_ufunc


class TestUfuncCacheTest(UfuncCacheTest):

    def test_direct_ufunc_cache(self, **kwargs):
        new_ufunc, cached_ufunc = self.check_ufunc_cache(
            "direct_ufunc_cache_usecase", n_overloads=2, **kwargs)
        # Test the cached and original versions
        inp = np.random.random(10).astype(np.float64)
        np.testing.assert_equal(new_ufunc(inp), cached_ufunc(inp))
        inp = np.arange(10, dtype=np.intp)
        np.testing.assert_equal(new_ufunc(inp), cached_ufunc(inp))

    def test_direct_ufunc_cache_objmode(self):
        self.test_direct_ufunc_cache(forceobj=True)

    def test_indirect_ufunc_cache(self):
        new_ufunc, cached_ufunc = self.check_ufunc_cache(
            "indirect_ufunc_cache_usecase", n_overloads=3)
        # Test the cached and original versions
        inp = np.random.random(10).astype(np.float64)
        np.testing.assert_equal(new_ufunc(inp), cached_ufunc(inp))
        inp = np.arange(10, dtype=np.intp)
        np.testing.assert_equal(new_ufunc(inp), cached_ufunc(inp))


class TestDUfuncCacheTest(UfuncCacheTest):
    def check_dufunc_usecase(self, usecase_name):

        mod = self.import_module()
        usecase = getattr(mod, usecase_name)
        # Create dufunc
        with self.capture_cache_log() as out:
            ufunc = usecase()
        self.check_cache_saved(out.getvalue(), count=0)
        # Compile & cache
        with self.capture_cache_log() as out:
            ufunc(np.arange(10))
        self.check_cache_saved(out.getvalue(), count=1)
        self.check_cache_loaded(out.getvalue(), count=0)
        # Use cached
        with self.capture_cache_log() as out:
            ufunc = usecase()
            ufunc(np.arange(10))
        self.check_cache_loaded(out.getvalue(), count=1)

    def test_direct_dufunc_cache(self):
        # We don't test for objmode because DUfunc don't support it.
        self.check_dufunc_usecase('direct_dufunc_cache_usecase')

    def test_indirect_dufunc_cache(self):
        self.check_dufunc_usecase('indirect_dufunc_cache_usecase')


class TestGUFuncCacheTest(UfuncCacheTest):

    def test_direct_gufunc_cache(self, **kwargs):
        new_ufunc, cached_ufunc = self.check_ufunc_cache(
            "direct_gufunc_cache_usecase", n_overloads=2, **kwargs)
        # Test the cached and original versions
        inp = np.random.random(10).astype(np.float64)
        np.testing.assert_equal(new_ufunc(inp), cached_ufunc(inp))
        inp = np.arange(10, dtype=np.intp)
        np.testing.assert_equal(new_ufunc(inp), cached_ufunc(inp))

    def test_direct_gufunc_cache_objmode(self):
        self.test_direct_gufunc_cache(forceobj=True)

    def test_indirect_gufunc_cache(self):
        new_ufunc, cached_ufunc = self.check_ufunc_cache(
            "indirect_gufunc_cache_usecase", n_overloads=3)
        # Test the cached and original versions
        inp = np.random.random(10).astype(np.float64)
        np.testing.assert_equal(new_ufunc(inp), cached_ufunc(inp))
        inp = np.arange(10, dtype=np.intp)
        np.testing.assert_equal(new_ufunc(inp), cached_ufunc(inp))


if __name__ == '__main__':
    unittest.main()
