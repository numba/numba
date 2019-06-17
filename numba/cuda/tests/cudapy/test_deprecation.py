from __future__ import print_function, absolute_import

import warnings
from contextlib import contextmanager

from numba.tests.support import override_config, TestCase
from numba.cuda.testing import skip_on_cudasim
from numba import unittest_support as unittest
from numba import cuda, types
from numba.cuda.testing import SerialMixin


@skip_on_cudasim("Skipped on simulator")
class TestCudaDebugInfo(SerialMixin, TestCase):
    """Tests features that will be deprecated
    """
    @contextmanager
    def assert_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            yield w

    def test_autotune(self):
        @cuda.jit("(int32[:],)")
        def foo(xs):
            xs[0] = 1

        with self.assert_deprecation_warning() as w:
            foo.autotune
            assert len(w) == 1
            assert issubclass(w[-1].category, DeprecationWarning)
            assert ".autotune" in str(w[-1].message)

        with self.assert_deprecation_warning() as w:
            foo.occupancy
            assert len(w) == 2
            assert issubclass(w[0].category, DeprecationWarning)
            assert ".occupancy" in str(w[0].message)
            assert issubclass(w[1].category, DeprecationWarning)
            assert ".autotune" in str(w[1].message)


if __name__ == '__main__':
    unittest.main()
