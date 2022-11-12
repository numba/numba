import os
import shutil
import subprocess
import sys

from numba.tests.support import temp_directory, import_dynamic
from numba.tests.test_caching import DispatcherCacheUsecasesTest


class TestCachingModifiedFilesBase(DispatcherCacheUsecasesTest):
    # the file with a main function that will call another one
    source_text_file1: str = ""
    # the file with a secondary function which is called from the main
    source_text_file2: str = ""
    # an alternative version of the file with the secondary function
    source_text_file2_alt: str = ""

    def setUp(self):
        self.tempdir = temp_directory('test_cache_file_modfiles2')
        self.cache_dir = os.path.join(self.tempdir, "__pycache__")

        self.modname = "file1"
        self.file1 = os.path.join(self.tempdir, 'file1.py')
        with open(self.file1, 'w') as fout:
            print(self.source_text_file1, file=fout)

        self.mod2name = "file2"
        self.file2 = os.path.join(self.tempdir, 'file2.py')
        with open(self.file2, 'w') as fout:
            print(self.source_text_file2, file=fout)

    def tearDown(self):
        sys.modules.pop(self.modname, None)
        sys.modules.pop(self.mod2name, None)
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

        self.modfile = self.file1
        self.cache_dir = os.path.join(self.tempdir, "__pycache__")

        # 1. execute original files once to populate cache
        self.run_fc_in_separate_process()
        # import function and verify cache is being used
        import sys
        sys.path.insert(0, self.tempdir)
        file1 = self.import_module()
        fc = file1.foo
        # First round of execution to populate cache
        self.assertPreciseEqual(fc(2), 2)
        # expected files: 1 index + 1 data for each function (if cached)
        self.check_pycache(2 + 2 * inner_cached)
        self.assertPreciseEqual(fc(2.5), 2.5)
        # expected files: 2 index, 2 data for each function (if cached)
        self.check_pycache(3 + 3 * inner_cached)
        self.check_hits(fc, 0, 2)

        # 2. Re-import module ane execute again, cached version should be used
        del fc
        del file1
        file1, file2 = self.import_modules(
            [self.modname, self.mod2name], [self.file1, self.file2]
        )
        fc = file1.foo
        self.assertPreciseEqual(fc(2), 2)
        # expected files: 2 index, 2 data for each function (if cached)
        self.check_pycache(3 + 3 * inner_cached)
        self.assertPreciseEqual(fc(2.5), 2.5)
        # expected files: 2 index, 2 data for each function (if cached)
        self.check_pycache(3 + 3 * inner_cached)
        self.check_hits(fc, 2, 0)

        # 3. modify file and reload
        with open(self.file2, 'w') as fout:
            print(self.source_text_file2_alt, file=fout)
        file1, file2 = self.import_modules(
            [self.modname, self.mod2name], [self.file1, self.file2]
        )
        fc = file1.foo

        # 4. Run again, results should change, cache should not be hit
        self.assertPreciseEqual(fc(2), 3)
        # 2 index, 2 data for foo function (2 from previous function2 versions
        # one of which is overwritten by the new version), 2 for function2.
        # If Function2 (inner function) is cached, then it has restarted
        # its cache after the change and it has 2 files
        # (1 new, and 1 stale which is out of the index which means that it
        # will be eventually overwritten)
        self.check_pycache(3 + 3 * inner_cached)
        self.assertPreciseEqual(fc(2.5), 3.5)
        # expected files: 2 index, 2 data for foo, 2 for function2 (if cached)
        self.check_pycache(3 + 3 * inner_cached)
        self.check_hits(fc, 0, 2)


class TestCachingModifiedFiles1(TestCachingModifiedFilesBase):
    # This class tests a dispatcher calling another dispatcher which later
    # changes. Both functions have cache=True

    source_text_file1 = """
from numba import njit
from file2 import function2
@njit(cache=True)
def foo(x):
    return function2(x)
"""
    source_text_file2 = """
from numba import njit
@njit(cache=True)
def function1(x):
    return x
@njit(cache=True)
def function2(x):
    return x
"""

    source_text_file2_alt = """
from numba import njit
@njit(cache=True)
def function1(x):
    return x + 1
@njit(cache=True)
def function2(x):
    return x + 1
"""

    def test_invalidation(self, ):
        self.execute_fc_and_change_it(inner_cached=True)


class TestCachingModifiedFiles2(TestCachingModifiedFilesBase):
    # This class tests a dispatcher calling another dispatcher which later
    # changes. Only the main function has cache=True
    source_text_file1 = """
from numba import njit
from file2 import function2
@njit(cache=True)
def foo(x):
    return function2(x)
"""
    source_text_file2 = """
from numba import njit
@njit()
def function1(x):
    return x
@njit()
def function2(x):
    return x
"""

    source_text_file2_alt = """
from numba import njit
@njit()
def function1(x):
    return x + 1
@njit()
def function2(x):
    return x + 1
    """

    def test_invalidation(self):
        # test cache is invalidated after source file modification
        self.execute_fc_and_change_it(inner_cached=False)


class TestCachingModifiedFiles3(TestCachingModifiedFilesBase):
    # This class tests a cfunc calling a dispatcher which later
    # changes. Only the main function has cache=True
    source_text_file1 = """
from numba import cfunc
from file2 import function2
@cfunc('float64(float64)',cache=True)
def foo(x):
    return function2(x)
"""
    source_text_file2 = """
from numba import njit
@njit()
def function1(x):
    return x
@njit()
def function2(x):
    return x
"""

    source_text_file2_alt = """
from numba import njit
@njit()
def function1(x):
    return x + 1
@njit()
def function2(x):
    return x + 1
    """

    def test_invalidation(self):
        # test cache is invalidated after source file modification
        self.execute_fc_and_change_it(inner_cached=False)

    def check_hits(self, func, *, hits, misses=None):
        # check_hits needs to be overriden because dispatchers use Counters
        # but Cfuncs use plain ints to keep track of misses
        st = func.stats
        self.assertEqual(st.cache_hits, hits, st.cache_hits)
        if misses is not None:
            self.assertEqual(st.cache_misses, misses,
                             st.cache_misses)

    def execute_fc_and_change_it(self, inner_cached):
        # test cache is invalidated after source file modification
        # inner_cached: boolean, whether the inner function is cached. This
        # changes the number of cache files that are expected

        # this method needs to be overriden in this class, because CFuncs
        # have only one signature, which means the number of cache files
        # and the number of hits and misses are different. Structurally
        # the code is identical

        self.modname = "file1"
        self.modfile = self.file1
        self.cache_dir = os.path.join(self.tempdir, "__pycache__")

        # 1. execute original files once to populate cache
        self.run_fc_in_separate_process()
        # import function and verify cache is being used
        import sys
        sys.path.insert(0, self.tempdir)
        file1 = self.import_module()
        fc = file1.foo
        # First round of execution to populate cache
        self.assertPreciseEqual(fc(2), 2)
        # expected files: 1 index + 1 data for each function
        self.check_pycache(2 + 2 * inner_cached)
        self.assertPreciseEqual(fc(2.5), 2.5)
        # expected files: 2 index, 2 data for each function
        self.check_pycache(2 + 2 * inner_cached)
        self.check_hits(fc, hits=1, misses=0)

        # 2. Re-import module ane execute again, cached version should be used
        del fc
        del file1
        file1, file2 = self.import_modules(
            ["file1", "file2"], [self.file1, self.file2]
        )
        fc = file1.foo
        self.assertPreciseEqual(fc(2), 2)
        # expected files: 2 index, 2 data for each function
        self.check_pycache(2 + 2 * inner_cached)
        self.assertPreciseEqual(fc(2.5), 2.5)
        # expected files: 2 index, 2 data for each function
        self.check_pycache(2 + 2 * inner_cached)
        self.check_hits(fc, hits=1, misses=0)

        # 3. modify file and reload
        self.file2_alt = os.path.join(self.tempdir, 'file2.py')
        with open(self.file2_alt, 'w') as fout:
            print(self.source_text_file2_alt, file=fout)

        file1, file2 = self.import_modules(
            ["file1", "file2"], [self.file1, self.file2]
        )
        fc = file1.foo
        # 4. Run again, results should change, cache should not be hit
        self.assertPreciseEqual(fc(2), 3)
        # 2 index, 2 data for foo function (2 from previous function2 versions
        # one of which is overwritten by the new version), 2 for function2.
        # Function2 has restarted its cache after the change
        # and it has 2 files (1 new, 1 stale but out of the index
        # which will be eventually overwriten)
        self.check_pycache(2 + 2 * inner_cached)
        self.assertPreciseEqual(fc(2.5), 3.5)
        # expected files: 2 index, 2 data for foo, 2 for function2
        self.check_pycache(2 + 2 * inner_cached)
        self.check_hits(fc, hits=0, misses=1)


class TestCachingModifiedFiles4(TestCachingModifiedFilesBase):
    # This class tests a dispatcher calling a user-defined overload which later
    # changes.
    source_text_file1 = """
from numba import njit
from file2 import function2
@njit(cache=True)
def foo(x):
    return function2(x)
"""
    source_text_file2 = """
from numba import njit
from numba.core.extending import overload
@njit()
def function1(x):
    return x

def function2(x):
    return x

@overload(function2)
def f2_ovrl(x):

    def f2_impl(x):
        return x

    return f2_impl
"""

    source_text_file2_alt = """
from numba import njit
from numba.core.extending import overload
@njit()
def function1(x):
    return x + 1

def function2(x):
    return x + 1

@overload(function2)
def f2_ovrl(x):

    def f2_impl(x):
        return x + 1

    return f2_impl
    """

    def test_invalidation(self):
        # test cache is invalidated after source file modification
        self.execute_fc_and_change_it(inner_cached=False)


class TestCachingModifiedFiles5(TestCachingModifiedFilesBase):
    # This class tests a user-defined overload calling a  dispatcher which later
    # changes.
    source_text_file1 = """
from numba.core.extending import overload
from numba import njit
from file2 import function2

def bar(x):
    raise NotImplementedError

@overload(bar)
def ovrl_bar(x):

    def impl_bar(x):
        return function2(x)

    return impl_bar

@njit(cache=True)
def foo(x):
    return bar(x)
"""
    source_text_file2 = """
from numba import njit
@njit()
def function1(x):
    return x
@njit()
def function2(x):
    return x
    """

    source_text_file2_alt = """
from numba import njit
@njit()
def function1(x):
    return x + 1
@njit()
def function2(x):
    return x + 1
        """

    def test_invalidation(self):
        # test cache is invalidated after source file modification
        self.execute_fc_and_change_it(inner_cached=False)


class TestCachingModifiedFiles6(TestCachingModifiedFilesBase):
    # This class tests a user-defined overload calling a  dispatcher which later
    # changes.
    source_text_file1 = """
from numba.core.extending import overload
from numba import njit
from file2 import function2

def bar(x):
    raise NotImplementedError

@overload(bar)
def ovrl_bar(x):

    def impl_bar(x):
        return function2(x)

    return impl_bar

@njit(cache=True)
def foo(x):
    return bar(x)
"""
    source_text_file2 = """
from numba import njit
@njit()
def function1(x):
    return x
@njit(cache=True)
def function2(x):
    return x
    """

    source_text_file2_alt = """
from numba import njit
@njit()
def function1(x):
    return x + 1
@njit(cache=True)
def function2(x):
    return x + 1
        """

    def test_invalidation(self):
        # test cache is invalidated after source file modification
        self.execute_fc_and_change_it(inner_cached=True)
