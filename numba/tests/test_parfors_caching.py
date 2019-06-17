from __future__ import print_function, absolute_import, division

import os.path

import numpy as np

from .support import skip_parfors_unsupported
from .test_dispatcher import BaseCacheUsecasesTest


@skip_parfors_unsupported
class TestParForsCache(BaseCacheUsecasesTest):
    here = os.path.dirname(__file__)
    usecases_file = os.path.join(here, "parfors_cache_usecases.py")
    modname = "parfors_caching_test_fodder"

    def test_arrayexprs(self):
        mod = self.import_module()
        self.check_pycache(0)
        f = mod.arrayexprs
        ary = np.ones(10)
        self.assertPreciseEqual(f(ary), ary / ary.sum())

        dynamic_globals = [cres.library.has_dynamic_globals
                           for cres in f.overloads.values()]
        [cres] = f.overloads.values()
        self.assertEqual(dynamic_globals, [False])
        self.check_pycache(2)  # 1 index, 1 data

        self.run_in_separate_process()
