import sys
import types

import pkg_resources

from .support import TestCase


class TestEntrypoints(TestCase):
    """
    Test registration of init() functions from Numba extensions
    """

    def test_init_entrypoint(self):
        # loosely based on Pandas test from:
        #   https://github.com/pandas-dev/pandas/pull/27488

        # FIXME: Python 2 workaround because nonlocal doesn't exist
        counters = {'init': 0}

        def init_function():
            counters['init'] += 1

        mod = types.ModuleType("_test_numba_extension")
        mod.init_func = init_function

        try:
            # will remove this module at the end of the test
            sys.modules[mod.__name__] = mod

            # We are registering an entry point using the "numba" package
            # ("distribution" in pkg_resources-speak) itself, though these are
            # normally registered by other packages.
            dist = "numba"
            entrypoints = pkg_resources.get_entry_map(dist)
            my_entrypoint = pkg_resources.EntryPoint(
                "init", # name of entry point
                mod.__name__, # module with entry point object
                attrs=['init_func'], # name of entry point object
                dist=pkg_resources.get_distribution(dist)
            )
            entrypoints.setdefault('numba_extensions',
                                   {})['init'] = my_entrypoint

            import numba.entrypoints
            # Allow reinitialization
            numba.entrypoints._already_initialized = False

            numba.entrypoints.init_all()

            # was our init function called?
            self.assertEqual(counters['init'], 1)

            # ensure we do not initialize twice
            numba.entrypoints.init_all()
            self.assertEqual(counters['init'], 1)
        finally:
            # remove fake module
            if mod.__name__ in sys.modules:
                del sys.modules[mod.__name__]
