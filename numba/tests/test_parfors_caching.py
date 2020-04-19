import os.path

import numpy as np

from numba.tests.support import skip_parfors_unsupported
from .test_dispatcher import BaseCacheUsecasesTest


@skip_parfors_unsupported
class TestParForsCache(BaseCacheUsecasesTest):
    here = os.path.dirname(__file__)
    usecases_file = os.path.join(here, "parfors_cache_usecases.py")
    modname = "parfors_caching_test_fodder"

    def run_test(self, fname, num_funcs=1):
        mod = self.import_module()
        self.check_pycache(0)
        f = getattr(mod, fname)
        ary = np.ones(10)
        self.assertPreciseEqual(f(ary), f.py_func(ary))

        dynamic_globals = [cres.library.has_dynamic_globals
                           for cres in f.overloads.values()]
        [cres] = f.overloads.values()
        self.assertEqual(dynamic_globals, [False])
        # For each cached func, there're 2 entries (index + data)
        self.check_pycache(num_funcs * 2)

        self.run_in_separate_process()

    def test_arrayexprs(self):
        f = 'arrayexprs_case'
        self.run_test(f)

    def test_prange(self):
        f = 'prange_case'
        self.run_test(f)

    def test_caller(self):
        f = 'caller_case'
        # num_funcs=3 because, there's the `caller_case()` which calls
        # the `prange_case()` and `arrayexprs_case()`
        self.run_test(f, num_funcs=3)
