from __future__ import print_function
import itertools
from numba import unittest_support as unittest
from numba import types
from numba.typeconv.typeconv import TypeManager
from numba.typeconv import rules


class TestTypeConv(unittest.TestCase):

    def test_typeconv(self):
        tm = TypeManager()

        i32 = types.int32
        i64 = types.int64
        f32 = types.float32

        tm.set_promote(i32, i64)
        tm.set_unsafe_convert(i32, f32)

        sig = (i32, f32)
        ovs = [
            (i32, i32),
            (f32, f32),
            (i64, i64),
        ]

        # allow_unsafe = True => a conversion from i32 to f32 is chosen
        sel = tm.select_overload(sig, ovs, True)
        self.assertEqual(sel, 1)
        # allow_unsafe = False => no overload available
        with self.assertRaises(TypeError):
            sel = tm.select_overload(sig, ovs, False)

    def test_default(self):
        tm = rules.default_type_manager

        i16 = types.int16
        i32 = types.int32
        i64 = types.int64
        f32 = types.float32

        self.assertEqual(tm.check_compatible(i32, i64), 'promote')
        self.assertEqual(tm.check_compatible(i32, f32), 'unsafe')

        self.assertEqual(tm.check_compatible(i16, i64), 'promote')

        for ta, tb in itertools.product(types.number_domain,
                                        types.number_domain):
            if ta in types.complex_domain and tb not in types.complex_domain:
                continue
            self.assertTrue(tm.check_compatible(ta, tb) is not None,
                            msg="No cast from %s to %s" % (ta, tb))

    def test_overload1(self):
        tm = rules.default_type_manager

        i32 = types.int32
        i64 = types.int64

        sig = (i64, i32, i32)
        ovs = [
            (i32, i32, i32),
            (i64, i64, i64),
        ]
        # The first overload is unsafe, the second is safe => the second
        # is always chosen, regardless of allow_unsafe.
        self.assertEqual(tm.select_overload(sig, ovs, True), 1)
        self.assertEqual(tm.select_overload(sig, ovs, False), 1)

    def test_overload2(self):
        tm = rules.default_type_manager

        i16 = types.int16
        i32 = types.int32
        i64 = types.int64

        sig = (i32, i16, i32)
        ovs = [
            # Three promotes
            (i64, i64, i64),
            # One promotes, two exact types
            (i32, i32, i32),
            # Two unsafe converts, one exact type
            (i16, i16, i16),
        ]
        self.assertEqual(tm.select_overload(sig, ovs, allow_unsafe=False), 1)
        self.assertEqual(tm.select_overload(sig, ovs, allow_unsafe=True), 1)

        # The same in reverse order
        ovs.reverse()
        self.assertEqual(tm.select_overload(sig, ovs, allow_unsafe=False), 1)
        self.assertEqual(tm.select_overload(sig, ovs, allow_unsafe=True), 1)

    def test_overload3(self):
        # Promotes should be preferred over safe converts
        tm = rules.default_type_manager

        i32 = types.int32
        i64 = types.int64
        f64 = types.float64

        sig = (i32, i32)
        ovs = [
            # Two promotes
            (i64, i64),
            # Two safe converts
            (f64, f64),
        ]
        self.assertEqual(tm.select_overload(sig, ovs, allow_unsafe=False), 0)
        self.assertEqual(tm.select_overload(sig, ovs, allow_unsafe=True), 0)

        # The same in reverse order
        ovs.reverse()
        self.assertEqual(tm.select_overload(sig, ovs, allow_unsafe=False), 1)
        self.assertEqual(tm.select_overload(sig, ovs, allow_unsafe=True), 1)


if __name__ == '__main__':
    unittest.main()
