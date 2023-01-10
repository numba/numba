import unittest
import numpy as np

from numba.tests.support import TestCase
from numba import typeof
from numba.core import types
from numba.typed import List, Dict

NB_TYPES = [
    types.Array, types.NestedArray, types.int64, types.float64,
    types.unicode_type, types.Record, types.UnicodeCharSeq, types.UniTuple,
    types.List, types.Tuple, types.DictType, types.ListType, types.Set,
    types.bool_
]


class TestRepr(TestCase):
    def check_repr(self, val):
        ty = typeof(val)
        tys_ns = {ty.__name__: ty for ty in NB_TYPES if hasattr(ty, "__name__")}
        tys_ns.update({ty.name: ty for ty in NB_TYPES if hasattr(ty, "name")})
        ty2 = eval(repr(ty), tys_ns)
        self.assertEqual(ty, ty2)

    def test_types(self):
        # define some values for the test cases
        rec_dtype = [("a", "f8"), ("b", "U8"), ("c", "i8", (2, 3))]
        nb_dict = Dict()
        nb_dict['a'] = 1
        # tests cases
        val_types_cases = [
            1,
            1.2,
            True,
            "a",
            (1, 2),
            (1, "a"),
            [1, "a"],
            ([1, "a"], [2, "b"]),
            ((1, 2), (3, "b")),
            ((1, 2), (3, [1, 2])),
            np.ones(3),
            np.array([(1, "a", np.ones((2, 3)))], dtype=rec_dtype),
            nb_dict,
            List([1, 2]),
            {1, 2},
        ]
        for val in val_types_cases:
            self.check_repr(val)

    def test_fail(self):
        self.assertTrue(False)

if __name__ == '__main__':
    unittest.main()
