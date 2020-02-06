import sys
from unittest import mock

try:
    from importlib import metadata as importlib_metadata
except ImportError:
    import importlib_metadata

from numba.tests.support import TestCase


class TestEntrypoints(TestCase):
    """
    Test registration of init() functions from Numba extensions
    """

    def test_init_entrypoint(self):
        # loosely based on Pandas test from:
        #   https://github.com/pandas-dev/pandas/pull/27488

        mod = mock.Mock(__name__='_test_numba_extension')

        try:
            # will remove this module at the end of the test
            sys.modules[mod.__name__] = mod

            # We are registering an entry point using the "numba" package
            # ("distribution" in importlib-speak) itself, though these are
            # normally registered by other packages.
            my_entrypoint = importlib_metadata.EntryPoint(
                'init', '_test_numba_extension:init_func', 'numba_extensions',
            )

            with mock.patch.object(
                importlib_metadata,
                'entry_points',
                return_value={'numba_extensions': (my_entrypoint,)},
            ):

                from numba.core import entrypoints

                # Allow reinitialization
                entrypoints._already_initialized = False

                entrypoints.init_all()

                # was our init function called?
                mod.init_func.assert_called_once()

                # ensure we do not initialize twice
                entrypoints.init_all()
                mod.init_func.assert_called_once()
        finally:
            # remove fake module
            if mod.__name__ in sys.modules:
                del sys.modules[mod.__name__]
