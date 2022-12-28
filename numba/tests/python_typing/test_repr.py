import unittest
import numpy as np

from numba.tests.support import TestCase
from numba import typeof
from numba.core import types


class MyTestCase(TestCase):
    def check_repr(self, val, nb_tys):
        # tys are numba types that are expected in locals and globals
        ty = typeof(val)
        print(val, ty)
        tys_ns = {ty.__name__: ty for ty in nb_tys if hasattr(ty, "__name__")}
        tys_ns.update({ty.name: ty for ty in nb_tys if hasattr(ty, "name")})
        ty2 = eval(repr(ty), tys_ns)
        self.assertEqual(ty, ty2)

    def test_types(self):

        val_types_cases = [
            (1, [types.int64]),
            ('a', [types.unicode_type]),
            ((1, 2), [types.UniTuple, types.int64]),
            ((1, 'a'), [types.Tuple, types.int64, types.unicode_type]),
            ([1, 'a'], [types.List, types.int64, types.unicode_type]),
            (([1, 'a'], [2, 'b']), [types.UniTuple, types.List, types.int64, types.unicode_type]),
            (np.ones(3), [types.Array, types.float64]),
            (np.array([(1, 'a', np.ones((2, 3)))], dtype=[('a', 'f8'), ('b', 'U8'), ('c', 'i8', (2,3))]), [types.Array,types.NestedArray, types.int64, types.float64, types.unicode_type, types.Record, types.UnicodeCharSeq])
        ]
        for val, tys in val_types_cases:
            self.check_repr(val, tys)


if __name__ == '__main__':
    unittest.main()
