from __future__ import print_function, absolute_import

from numba import unittest_support as unittest
from numba import itanium_mangler
from numba import int32, int64, uint32, uint64, float32, float64
from numba.types import range_iter32_type


class TestItaniumManager(unittest.TestCase):
    def test_ident(self):
        got = itanium_mangler.mangle_identifier("apple")
        expect = "5apple"
        self.assertEqual(expect, got)

        got = itanium_mangler.mangle_identifier("ap_ple")
        expect = "6ap_ple"
        self.assertEqual(expect, got)

        got = itanium_mangler.mangle_identifier("apple213")
        expect = "8apple213"
        self.assertEqual(expect, got)

    def test_types(self):
        got = itanium_mangler.mangle_type(int32)
        expect = "i"
        self.assertEqual(expect, got)

        got = itanium_mangler.mangle_type(int64)
        expect = "x"
        self.assertEqual(expect, got)

        got = itanium_mangler.mangle_type(uint32)
        expect = "j"
        self.assertEqual(expect, got)

        got = itanium_mangler.mangle_type(uint64)
        expect = "y"
        self.assertEqual(expect, got)

        got = itanium_mangler.mangle_type(float32)
        expect = "f"
        self.assertEqual(expect, got)

        got = itanium_mangler.mangle_type(float64)
        expect = "d"
        self.assertEqual(expect, got)

    def test_function(self):
        got = itanium_mangler.mangle("what", [int32, float32])
        expect = "_Z4whatif"
        self.assertEqual(expect, got)

        got = itanium_mangler.mangle("a_little_brown_fox", [uint64,
                                                            uint32,
                                                            float64])
        expect = "_Z18a_little_brown_foxyjd"
        self.assertEqual(expect, got)

    def test_custom_type(self):
        got = itanium_mangler.mangle_type(range_iter32_type)
        name = str(range_iter32_type)
        expect = "u{n}{name}".format(n=len(name), name=name)
        self.assertEqual(expect, got)


if __name__ == '__main__':
    unittest.main()
