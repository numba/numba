import shutil
import sys
import os.path
import re
import subprocess

import numpy as np

from numba.tests.support import capture_cache_log, temp_directory, \
    import_dynamic
from numba.tests.test_caching import BaseCacheTest
from numba.core import config
import unittest


class UfuncCacheTest(BaseCacheTest):
    """
    Since the cache stats is not exposed by ufunc, we test by looking at the
    cache debug log.
    """
    _numba_parallel_test_ = False

    here = os.path.dirname(__file__)
    usecases_file = os.path.join(here, "cache_usecases.py")
    modname = "ufunc_caching_test_fodder"

    regex_data_saved = re.compile(r'\[cache\] data saved to')
    regex_index_saved = re.compile(r'\[cache\] index saved to')

    regex_data_loaded = re.compile(r'\[cache\] data loaded from')
    regex_index_loaded = re.compile(r'\[cache\] index loaded from')

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
        with capture_cache_log() as out:
            new_ufunc = usecase(**kwargs)
        cachelog = out.getvalue()
        self.check_cache_saved(cachelog, count=n_overloads)

        # Use cached version
        with capture_cache_log() as out:
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

    def test_direct_ufunc_cache_parallel(self):
        self.test_direct_ufunc_cache(target='parallel')

    def test_indirect_ufunc_cache(self, **kwargs):
        new_ufunc, cached_ufunc = self.check_ufunc_cache(
            "indirect_ufunc_cache_usecase", n_overloads=3, **kwargs)
        # Test the cached and original versions
        inp = np.random.random(10).astype(np.float64)
        np.testing.assert_equal(new_ufunc(inp), cached_ufunc(inp))
        inp = np.arange(10, dtype=np.intp)
        np.testing.assert_equal(new_ufunc(inp), cached_ufunc(inp))

    def test_indirect_ufunc_cache_parallel(self):
        self.test_indirect_ufunc_cache(target='parallel')


class TestDUfuncCacheTest(UfuncCacheTest):
    # Note: DUFunc doesn't support parallel target yet

    def check_dufunc_usecase(self, usecase_name):
        mod = self.import_module()
        usecase = getattr(mod, usecase_name)
        # Create dufunc
        with capture_cache_log() as out:
            ufunc = usecase()
        self.check_cache_saved(out.getvalue(), count=0)
        # Compile & cache
        with capture_cache_log() as out:
            ufunc(np.arange(10))
        self.check_cache_saved(out.getvalue(), count=1)
        self.check_cache_loaded(out.getvalue(), count=0)
        # Use cached
        with capture_cache_log() as out:
            ufunc = usecase()
            ufunc(np.arange(10))
        self.check_cache_loaded(out.getvalue(), count=1)

    def test_direct_dufunc_cache(self):
        # We don't test for objmode because DUfunc don't support it.
        self.check_dufunc_usecase('direct_dufunc_cache_usecase')

    def test_indirect_dufunc_cache(self):
        self.check_dufunc_usecase('indirect_dufunc_cache_usecase')


def _fix_raw_path(rstr):
    if config.IS_WIN32:
        rstr = rstr.replace(r'/', r'\\\\')
    return rstr


class TestGUfuncCacheTest(UfuncCacheTest):

    def test_filename_prefix(self):
        mod = self.import_module()
        usecase = getattr(mod, "direct_gufunc_cache_usecase")
        with capture_cache_log() as out:
            usecase()
        cachelog = out.getvalue()
        # find number filename with "guf-" prefix
        fmt1 = _fix_raw_path(r'/__pycache__/guf-{}')
        prefixed = re.findall(fmt1.format(self.modname), cachelog)
        fmt2 = _fix_raw_path(r'/__pycache__/{}')
        normal = re.findall(fmt2.format(self.modname), cachelog)
        # expecting 2 overloads
        self.assertGreater(len(normal), 2)
        # expecting equal number of wrappers and overloads cache entries
        self.assertEqual(len(normal), len(prefixed))

    def test_direct_gufunc_cache(self, **kwargs):
        # 2 cache entry for the 2 overloads
        # and 2 cache entry for the gufunc wrapper
        new_ufunc, cached_ufunc = self.check_ufunc_cache(
            "direct_gufunc_cache_usecase", n_overloads=2 + 2, **kwargs)
        # Test the cached and original versions
        inp = np.random.random(10).astype(np.float64)
        np.testing.assert_equal(new_ufunc(inp), cached_ufunc(inp))
        inp = np.arange(10, dtype=np.intp)
        np.testing.assert_equal(new_ufunc(inp), cached_ufunc(inp))

    def test_direct_gufunc_cache_objmode(self):
        self.test_direct_gufunc_cache(forceobj=True)

    def test_direct_gufunc_cache_parallel(self):
        self.test_direct_gufunc_cache(target='parallel')

    def test_indirect_gufunc_cache(self, **kwargs):
        # 3 cache entry for the 3 overloads
        # and no cache entry for the gufunc wrapper
        new_ufunc, cached_ufunc = self.check_ufunc_cache(
            "indirect_gufunc_cache_usecase", n_overloads=3, **kwargs)
        # Test the cached and original versions
        inp = np.random.random(10).astype(np.float64)
        np.testing.assert_equal(new_ufunc(inp), cached_ufunc(inp))
        inp = np.arange(10, dtype=np.intp)
        np.testing.assert_equal(new_ufunc(inp), cached_ufunc(inp))

    def test_indirect_gufunc_cache_parallel(self, **kwargs):
        self.test_indirect_gufunc_cache(target='parallel')


class TestCacheSpecificIssue(UfuncCacheTest):

    def run_in_separate_process(self, runcode):
        # Based on the same name util function in test_dispatcher but modified
        # to allow user to define what to run.
        code = """if 1:
            import sys

            sys.path.insert(0, %(tempdir)r)
            mod = __import__(%(modname)r)
            mod.%(runcode)s
            """ % dict(tempdir=self.tempdir, modname=self.modname,
                       runcode=runcode)

        popen = subprocess.Popen([sys.executable, "-c", code],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = popen.communicate()
        if popen.returncode != 0:
            raise AssertionError("process failed with code %s: stderr follows"
                                 "\n%s\n" % (popen.returncode, err.decode()))

    #
    # The following test issue #2198 that loading cached (g)ufunc first
    # bypasses some target context initialization.
    #

    def test_first_load_cached_ufunc(self):
        # ensure function is cached
        self.run_in_separate_process('direct_ufunc_cache_usecase()')
        # use the cached function
        # this will fail if the target context is not init'ed
        self.run_in_separate_process('direct_ufunc_cache_usecase()')

    def test_first_load_cached_gufunc(self):
        # ensure function is cached
        self.run_in_separate_process('direct_gufunc_cache_usecase()')
        # use the cached function
        # this will fail out if the target context is not init'ed
        self.run_in_separate_process('direct_gufunc_cache_usecase()')


class TestCachingModifiedFilesBase(BaseCacheTest):
    # the file with a main function that will call another one
    source_text_file1: str = ""
    # the file with a secondary function which is called from the main
    source_text_file2: str = ""
    # an alternative version of the file with the secondary function
    source_text_file2_alt: str = ""

    def setUp(self):
        self.tempdir = temp_directory('test_cache_file_modfiles2')
        self.cache_dir = os.path.join(self.tempdir, "__pycache__")

        self.file1 = os.path.join(self.tempdir, 'file1.py')
        with open(self.file1, 'w') as fout:
            print(self.source_text_file1, file=fout)

        self.file2 = os.path.join(self.tempdir, 'file2.py')
        with open(self.file2, 'w') as fout:
            print(self.source_text_file2, file=fout)

    def tearDown(self):
        sys.modules.pop(self.modname, None)
        sys.modules.pop("file2", None)
        sys.path.remove(self.tempdir)
        shutil.rmtree(self.tempdir)

    @staticmethod
    def import_modules(modnames, modfiles):
        # Import a fresh version of the test modules.  All jitted functions
        # in the test module will start anew and load overloads from
        # the on-disk cache if possible.

        assert len(modnames) == len(modfiles)
        # all modules must be removed first
        for modname in modnames:
            old = sys.modules.pop(modname, None)
            if old is not None:
                # Make sure cached bytecode is removed
                cached = [old.__cached__]
                for fn in cached:
                    try:
                        os.unlink(fn)
                    except FileNotFoundError:
                        pass

        mods = []
        for modname, modfile in zip(modnames, modfiles):
            mod = import_dynamic(modname)
            mods.append(mod)
        return mods

    def run_fc_in_separate_process(self):
        # Execute file1.py
        popen = subprocess.Popen([sys.executable, self.file1],
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
        out, err = popen.communicate()
        msg = f"stdout:\n{out.decode()}\n\nstderr:\n{err.decode()}"
        self.assertEqual(popen.returncode, 0, msg=msg)

        # Execute file2.py
        popen = subprocess.Popen([sys.executable, self.file2],
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
        out, err = popen.communicate()
        msg = f"stdout:\n{out.decode()}\n\nstderr:\n{err.decode()}"

    def execute_fc_and_change_it(self, inner_cached):
        # test cache is invalidated after source file modification
        # inner_cached: boolean, whether the inner function is cached. This
        # changes the number of cache files that are expected

        self.modname = "file1"
        self.modfile = self.file1
        self.cache_dir = os.path.join(self.tempdir, "__pycache__")

        # 1. execute original files once to populate cache
        self.run_fc_in_separate_process()
        # import function and verify cache is being used
        import sys
        sys.path.insert(0, self.tempdir)
        file1 = self.import_module()
        fc = file1.ufunc
        # First round of execution to populate cache
        self.assertPreciseEqual(fc(2), 2)
        self.assertPreciseEqual(fc(2.5), 2.5)
        # expected files: 2 index, 2 data for each function
        self.check_pycache(3 + 3 * inner_cached)

        # 2. Re-import module ane execute again, cached version should be used
        del fc
        del file1
        file1, file2 = self.import_modules(
            ["file1", "file2"], [self.file1, self.file2]
        )
        fc = file1.ufunc
        self.assertPreciseEqual(fc(2), 2)
        # expected files: 2 index, 2 data for each function
        self.check_pycache(3 + 3 * inner_cached)
        self.assertPreciseEqual(fc(2.5), 2.5)
        # expected files: 2 index, 2 data for each function
        self.check_pycache(3 + 3 * inner_cached)

        # 3. modify file and reload
        self.file2_alt = os.path.join(self.tempdir, 'file2.py')
        with open(self.file2_alt, 'w') as fout:
            print(self.source_text_file2_alt, file=fout)

        file1, file2 = self.import_modules(
            ["file1", "file2"], [self.file1, self.file2]
        )
        fc = file1.ufunc
        # 4. Run again, results should change, cache should not be hit
        self.assertPreciseEqual(fc(2), 3)
        # 2 index, 2 data for foo function (2 from previous function2 versions
        # one of which is overwritten by the new version), 2 for function2.
        # Function2 has restarted its cache after the change
        # and it has 2 files (1 new, 1 stale but out of the index
        # which will be eventually overwriten)
        self.check_pycache(3 + 3 * inner_cached)
        self.assertPreciseEqual(fc(2.5), 3.5)
        # expected files: 2 index, 2 data for foo, 2 for function2
        self.check_pycache(3 + 3 * inner_cached)

    def check_pycache(self, n):
        c = self.cache_contents()
        self.assertEqual(len(c), n, c)


class TestCachingModifiedFiles(TestCachingModifiedFilesBase):
    # This class tests a dispatcher calling another dispatcher which later
    # changes. Both functions have cache=True

    source_text_file1 = """
from numba import vectorize
from file2 import indirect_ufunc_core2

@vectorize(["intp(intp)", "float64(float64)"], cache=True)
def ufunc(inp):
    return indirect_ufunc_core2(inp)
"""
    source_text_file2 = """
from numba import njit
@njit(cache=True)
def indirect_ufunc_core1(inp):
    return inp

@njit(cache=True)
def indirect_ufunc_core2(inp):
    return inp
"""

    source_text_file2_alt = """
from numba import njit
@njit(cache=True)
def indirect_ufunc_core1(inp):
    return inp + 1

@njit(cache=True)
def indirect_ufunc_core2(inp):
    return inp + 1
"""

    def test_invalidation(self, ):
        self.execute_fc_and_change_it(inner_cached=True)


class TestCachingModifiedFiles2(TestCachingModifiedFilesBase):
    # This class tests a dispatcher calling another dispatcher which later
    # changes. Only outer function has cache=True

    source_text_file1 = """
from numba import vectorize
from file2 import indirect_ufunc_core2

@vectorize(["intp(intp)", "float64(float64)"], cache=True)
def ufunc(inp):
    return indirect_ufunc_core2(inp)
"""
    source_text_file2 = """
from numba import njit
@njit
def indirect_ufunc_core1(inp):
    return inp

@njit
def indirect_ufunc_core2(inp):
    return inp
"""

    source_text_file2_alt = """
from numba import njit
@njit
def indirect_ufunc_core1(inp):
    return inp + 1

@njit
def indirect_ufunc_core2(inp):
    return inp + 1
"""

    def test_invalidation(self, ):
        self.execute_fc_and_change_it(inner_cached=False)


if __name__ == '__main__':
    unittest.main()
