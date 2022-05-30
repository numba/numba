import multiprocessing
import os
import shutil
import unittest
import warnings

from numba import cuda
from numba.core.errors import NumbaWarning
from numba.cuda.testing import CUDATestCase, skip_on_cudasim
from numba.tests.support import SerialMixin
from numba.tests.test_caching import (DispatcherCacheUsecasesTest,
                                      skip_bad_access)


@skip_on_cudasim('Simulator does not implement caching')
class CUDACachingTest(SerialMixin, DispatcherCacheUsecasesTest):
    here = os.path.dirname(__file__)
    usecases_file = os.path.join(here, "cache_usecases.py")
    modname = "cuda_caching_test_fodder"

    def setUp(self):
        DispatcherCacheUsecasesTest.setUp(self)
        CUDATestCase.setUp(self)

    def tearDown(self):
        CUDATestCase.tearDown(self)
        DispatcherCacheUsecasesTest.tearDown(self)

    def test_caching(self):
        self.check_pycache(0)
        mod = self.import_module()
        self.check_pycache(0)

        f = mod.add_usecase
        self.assertPreciseEqual(f(2, 3), 6)
        self.check_pycache(2)  # 1 index, 1 data
        self.assertPreciseEqual(f(2.5, 3), 6.5)
        self.check_pycache(3)  # 1 index, 2 data
        self.check_hits(f.func, 0, 2)

        f = mod.record_return_aligned
        rec = f(mod.aligned_arr, 1)
        self.assertPreciseEqual(tuple(rec), (2, 43.5))

        f = mod.record_return_packed
        rec = f(mod.packed_arr, 1)
        self.assertPreciseEqual(tuple(rec), (2, 43.5))
        self.check_pycache(6)  # 2 index, 4 data
        self.check_hits(f.func, 0, 2)

        # Check the code runs ok from another process
        self.run_in_separate_process()

    def test_no_caching(self):
        mod = self.import_module()

        f = mod.add_nocache_usecase
        self.assertPreciseEqual(f(2, 3), 6)
        self.check_pycache(0)

    def test_closure(self):
        mod = self.import_module()

        with warnings.catch_warnings():
            warnings.simplefilter('error', NumbaWarning)

            f = mod.closure1
            self.assertPreciseEqual(f(3), 6) # 3 + 3 = 6
            f = mod.closure2
            self.assertPreciseEqual(f(3), 8) # 3 + 5 = 8
            f = mod.closure3
            self.assertPreciseEqual(f(3), 10) # 3 + 7 = 10
            f = mod.closure4
            self.assertPreciseEqual(f(3), 12) # 3 + 9 = 12
            self.check_pycache(5) # 1 nbi, 4 nbc

    def test_cache_reuse(self):
        mod = self.import_module()
        mod.add_usecase(2, 3)
        mod.add_usecase(2.5, 3.5)
        mod.outer_uncached(2, 3)
        mod.outer(2, 3)
        mod.record_return_packed(mod.packed_arr, 0)
        mod.record_return_aligned(mod.aligned_arr, 1)
        mod.simple_usecase_caller(2)
        mtimes = self.get_cache_mtimes()
        # Two signatures compiled
        self.check_hits(mod.add_usecase.func, 0, 2)

        mod2 = self.import_module()
        self.assertIsNot(mod, mod2)
        f = mod2.add_usecase
        f(2, 3)
        self.check_hits(f.func, 1, 0)
        f(2.5, 3.5)
        self.check_hits(f.func, 2, 0)

        # The files haven't changed
        self.assertEqual(self.get_cache_mtimes(), mtimes)

        self.run_in_separate_process()
        self.assertEqual(self.get_cache_mtimes(), mtimes)

    def test_cache_invalidate(self):
        mod = self.import_module()
        f = mod.add_usecase
        self.assertPreciseEqual(f(2, 3), 6)

        # This should change the functions' results
        with open(self.modfile, "a") as f:
            f.write("\nZ = 10\n")

        mod = self.import_module()
        f = mod.add_usecase
        self.assertPreciseEqual(f(2, 3), 15)

    def test_recompile(self):
        # Explicit call to recompile() should overwrite the cache
        mod = self.import_module()
        f = mod.add_usecase
        self.assertPreciseEqual(f(2, 3), 6)

        mod = self.import_module()
        f = mod.add_usecase
        mod.Z = 10
        self.assertPreciseEqual(f(2, 3), 6)
        f.func.recompile()
        self.assertPreciseEqual(f(2, 3), 15)

        # Freshly recompiled version is re-used from other imports
        mod = self.import_module()
        f = mod.add_usecase
        self.assertPreciseEqual(f(2, 3), 15)

    def test_same_names(self):
        # Function with the same names should still disambiguate
        mod = self.import_module()
        f = mod.renamed_function1
        self.assertPreciseEqual(f(2), 4)
        f = mod.renamed_function2
        self.assertPreciseEqual(f(2), 8)

    def _test_pycache_fallback(self):
        """
        With a disabled __pycache__, test there is a working fallback
        (e.g. on the user-wide cache dir)
        """
        mod = self.import_module()
        f = mod.add_usecase
        # Remove this function's cache files at the end, to avoid accumulation
        # across test calls.
        self.addCleanup(shutil.rmtree, f.func.stats.cache_path,
                        ignore_errors=True)

        self.assertPreciseEqual(f(2, 3), 6)
        # It's a cache miss since the file was copied to a new temp location
        self.check_hits(f.func, 0, 1)

        # Test re-use
        mod2 = self.import_module()
        f = mod2.add_usecase
        self.assertPreciseEqual(f(2, 3), 6)
        self.check_hits(f.func, 1, 0)

        # The __pycache__ is empty (otherwise the test's preconditions
        # wouldn't be met)
        self.check_pycache(0)

    @skip_bad_access
    @unittest.skipIf(os.name == "nt",
                     "cannot easily make a directory read-only on Windows")
    def test_non_creatable_pycache(self):
        # Make it impossible to create the __pycache__ directory
        old_perms = os.stat(self.tempdir).st_mode
        os.chmod(self.tempdir, 0o500)
        self.addCleanup(os.chmod, self.tempdir, old_perms)

        self._test_pycache_fallback()

    @skip_bad_access
    @unittest.skipIf(os.name == "nt",
                     "cannot easily make a directory read-only on Windows")
    def test_non_writable_pycache(self):
        # Make it impossible to write to the __pycache__ directory
        pycache = os.path.join(self.tempdir, '__pycache__')
        os.mkdir(pycache)
        old_perms = os.stat(pycache).st_mode
        os.chmod(pycache, 0o500)
        self.addCleanup(os.chmod, pycache, old_perms)

        self._test_pycache_fallback()


@skip_on_cudasim('Simulator does not implement caching')
class CUDAAndCPUCachingTest(SerialMixin, DispatcherCacheUsecasesTest):
    here = os.path.dirname(__file__)
    usecases_file = os.path.join(here, "cache_with_cpu_usecases.py")
    modname = "cuda_and_cpu_caching_test_fodder"

    def setUp(self):
        DispatcherCacheUsecasesTest.setUp(self)
        CUDATestCase.setUp(self)

    def tearDown(self):
        CUDATestCase.tearDown(self)
        DispatcherCacheUsecasesTest.tearDown(self)

    def test_cpu_and_cuda_targets(self):
        # The same function jitted for CPU and CUDA targets should maintain
        # separate caches for each target.
        self.check_pycache(0)
        mod = self.import_module()
        self.check_pycache(0)

        f_cpu = mod.assign_cpu
        f_cuda = mod.assign_cuda
        self.assertPreciseEqual(f_cpu(5), 5)
        self.check_pycache(2)  # 1 index, 1 data
        self.assertPreciseEqual(f_cuda(5), 5)
        self.check_pycache(3)  # 1 index, 2 data

        self.check_hits(f_cpu.func, 0, 1)
        self.check_hits(f_cuda.func, 0, 1)

        self.assertPreciseEqual(f_cpu(5.5), 5.5)
        self.check_pycache(4)  # 1 index, 3 data
        self.assertPreciseEqual(f_cuda(5.5), 5.5)
        self.check_pycache(5)  # 1 index, 4 data

        self.check_hits(f_cpu.func, 0, 2)
        self.check_hits(f_cuda.func, 0, 2)

    def test_cpu_and_cuda_reuse(self):
        # Existing cache files for the CPU and CUDA targets are reused.
        mod = self.import_module()
        mod.assign_cpu(5)
        mod.assign_cpu(5.5)
        mod.assign_cuda(5)
        mod.assign_cuda(5.5)

        mtimes = self.get_cache_mtimes()

        # Two signatures compiled
        self.check_hits(mod.assign_cpu.func, 0, 2)
        self.check_hits(mod.assign_cuda.func, 0, 2)

        mod2 = self.import_module()
        self.assertIsNot(mod, mod2)
        f_cpu = mod2.assign_cpu
        f_cuda = mod2.assign_cuda

        f_cpu(2)
        self.check_hits(f_cpu.func, 1, 0)
        f_cpu(2.5)
        self.check_hits(f_cpu.func, 2, 0)
        f_cuda(2)
        self.check_hits(f_cuda.func, 1, 0)
        f_cuda(2.5)
        self.check_hits(f_cuda.func, 2, 0)

        # The files haven't changed
        self.assertEqual(self.get_cache_mtimes(), mtimes)

        self.run_in_separate_process()
        self.assertEqual(self.get_cache_mtimes(), mtimes)


def get_different_cc_gpus():
    # Find two GPUs with different Compute Capabilities and return them as a
    # tuple. If two GPUs with distinct Compute Capabilities cannot be found,
    # then None is returned.
    first_gpu = cuda.gpus[0]
    with first_gpu:
        first_cc = cuda.current_context().device.compute_capability

    for gpu in cuda.gpus[1:]:
        with gpu:
            cc = cuda.current_context().device.compute_capability
            if cc != first_cc:
                return (first_gpu, gpu)

    return None


@skip_on_cudasim('Simulator does not implement caching')
class TestMultiCCCaching(SerialMixin, DispatcherCacheUsecasesTest):
    here = os.path.dirname(__file__)
    usecases_file = os.path.join(here, "cache_usecases.py")
    modname = "cuda_multi_cc_caching_test_fodder"

    def setUp(self):
        DispatcherCacheUsecasesTest.setUp(self)
        CUDATestCase.setUp(self)

    def tearDown(self):
        CUDATestCase.tearDown(self)
        DispatcherCacheUsecasesTest.tearDown(self)

    def test_cache(self):
        gpus = get_different_cc_gpus()
        if not gpus:
            self.skipTest('Need two different CCs for multi-CC cache test')

        self.check_pycache(0)
        mod = self.import_module()
        self.check_pycache(0)

        # Populate the cache with the first GPU
        with gpus[0]:
            f = mod.add_usecase
            self.assertPreciseEqual(f(2, 3), 6)
            self.check_pycache(2)  # 1 index, 1 data
            self.assertPreciseEqual(f(2.5, 3), 6.5)
            self.check_pycache(3)  # 1 index, 2 data
            self.check_hits(f.func, 0, 2)

            f = mod.record_return_aligned
            rec = f(mod.aligned_arr, 1)
            self.assertPreciseEqual(tuple(rec), (2, 43.5))

            f = mod.record_return_packed
            rec = f(mod.packed_arr, 1)
            self.assertPreciseEqual(tuple(rec), (2, 43.5))
            self.check_pycache(6)  # 2 index, 4 data
            self.check_hits(f.func, 0, 2)

        # Run with the second GPU - under present behaviour this doesn't
        # further populate the cache.
        with gpus[1]:
            f = mod.add_usecase
            self.assertPreciseEqual(f(2, 3), 6)
            self.check_pycache(6)  # cache unchanged
            self.assertPreciseEqual(f(2.5, 3), 6.5)
            self.check_pycache(6)  # cache unchanged
            self.check_hits(f.func, 0, 2)

            f = mod.record_return_aligned
            rec = f(mod.aligned_arr, 1)
            self.assertPreciseEqual(tuple(rec), (2, 43.5))

            f = mod.record_return_packed
            rec = f(mod.packed_arr, 1)
            self.assertPreciseEqual(tuple(rec), (2, 43.5))
            self.check_pycache(6)  # cache unchanged
            self.check_hits(f.func, 0, 2)

        # Run in a separate module with the second GPU - this populates the
        # cache for the second CC.
        mod2 = self.import_module()
        self.assertIsNot(mod, mod2)

        with gpus[1]:
            f = mod2.add_usecase
            self.assertPreciseEqual(f(2, 3), 6)
            self.check_pycache(7)  # 2 index, 5 data
            self.assertPreciseEqual(f(2.5, 3), 6.5)
            self.check_pycache(8)  # 2 index, 6 data
            self.check_hits(f.func, 0, 2)

            f = mod2.record_return_aligned
            rec = f(mod.aligned_arr, 1)
            self.assertPreciseEqual(tuple(rec), (2, 43.5))

            f = mod2.record_return_packed
            rec = f(mod.packed_arr, 1)
            self.assertPreciseEqual(tuple(rec), (2, 43.5))
            self.check_pycache(10)  # 2 index, 8 data
            self.check_hits(f.func, 0, 2)


def child_initializer():
    # Disable occupancy and implicit copy warnings in processes in a
    # multiprocessing pool.
    from numba.core import config
    config.CUDA_LOW_OCCUPANCY_WARNINGS = 0
    config.CUDA_WARN_ON_IMPLICIT_COPY = 0


@skip_on_cudasim('Simulator does not implement caching')
class TestMultiprocessCache(SerialMixin, DispatcherCacheUsecasesTest):

    # Nested multiprocessing.Pool raises AssertionError:
    # "daemonic processes are not allowed to have children"
    _numba_parallel_test_ = False

    here = os.path.dirname(__file__)
    usecases_file = os.path.join(here, "cache_usecases.py")
    modname = "cuda_mp_caching_test_fodder"

    def setUp(self):
        DispatcherCacheUsecasesTest.setUp(self)
        CUDATestCase.setUp(self)

    def tearDown(self):
        CUDATestCase.tearDown(self)
        DispatcherCacheUsecasesTest.tearDown(self)

    def test_multiprocessing(self):
        # Check caching works from multiple processes at once (#2028)
        mod = self.import_module()
        # Calling a pure Python caller of the JIT-compiled function is
        # necessary to reproduce the issue.
        f = mod.simple_usecase_caller
        n = 3
        try:
            ctx = multiprocessing.get_context('spawn')
        except AttributeError:
            ctx = multiprocessing

        pool = ctx.Pool(n, child_initializer)

        try:
            res = sum(pool.imap(f, range(n)))
        finally:
            pool.close()
        self.assertEqual(res, n * (n - 1) // 2)
