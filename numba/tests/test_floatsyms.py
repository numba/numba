from __future__ import print_function
import numba.unittest_support as unittest
from numba.compiler import compile_isolated
from numba import types


class TestFloatSymbols(unittest.TestCase):
    """
    Test ftol symbols on windows
    """

    def _test_template(self, realty, intty):
        def cast(x):
            y = x
            return y

        cres = compile_isolated(cast, args=[realty], return_type=intty)
        self.assertAlmostEqual(cres.entry_point(1.), 1)

    def test_float64_to_int64(self):
        self._test_template(types.float64, types.int64)

    def test_float64_to_uint64(self):
        self._test_template(types.float64, types.uint64)

    def test_float64_to_int32(self):
        self._test_template(types.float64, types.int32)

    def test_float64_to_uint32(self):
        self._test_template(types.float64, types.uint32)

    def test_float32_to_int64(self):
        self._test_template(types.float32, types.int64)

    def test_float32_to_uint64(self):
        self._test_template(types.float32, types.uint64)

    def test_float32_to_int32(self):
        self._test_template(types.float32, types.int32)

    def test_float32_to_uint32(self):
        self._test_template(types.float32, types.uint32)


if __name__ == '__main__':
    unittest.main()
