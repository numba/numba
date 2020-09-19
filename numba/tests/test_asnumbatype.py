"""
Tests for the as_numba_type() machinery.
"""
import typing as pt
import unittest

import numpy as np
import typing_extensions as pt_ext
from numba.core import types
from numba.core.errors import TypingError
from numba.core.typing.asnumbatype import AsNumbaTypeRegistry, as_numba_type
from numba.core.typing.typeof import typeof
from numba.experimental.jitclass import jitclass
from numba.np import numpy_support
from numba.tests.support import TestCase


class TestAsNumbaType(TestCase):

    int_nb_type = typeof(0)
    float_nb_type = typeof(0.0)
    complex_nb_type = typeof(complex(0))
    str_nb_type = typeof("numba")
    bool_nb_type = typeof(True)
    none_nb_type = typeof(None)

    def test_simple_types(self):
        self.assertEqual(as_numba_type(int), self.int_nb_type)
        self.assertEqual(as_numba_type(float), self.float_nb_type)
        self.assertEqual(as_numba_type(complex), self.complex_nb_type)
        self.assertEqual(as_numba_type(str), self.str_nb_type)
        self.assertEqual(as_numba_type(bool), self.bool_nb_type)
        self.assertEqual(as_numba_type(type(None)), self.none_nb_type)

    def test_numba_types(self):
        numba_types = [
            types.intp,
            types.boolean,
            types.ListType(types.float64),
            types.DictType(
                types.intp, types.Tuple([types.float32, types.float32])
            ),
        ]

        for ty in numba_types:
            self.assertEqual(as_numba_type(ty), ty)

    def test_numpy_types(self):
        numpy_types = [
            np.float32,
            np.bool_,
            np.int16,
        ]

        for ty in numpy_types:
            self.assertEqual(as_numba_type(ty), numpy_support.from_dtype(ty))

    def test_single_containers(self):
        self.assertEqual(
            as_numba_type(pt.List[float]),
            types.ListType(self.float_nb_type),
        )
        self.assertEqual(
            as_numba_type(pt.Dict[float, str]),
            types.DictType(self.float_nb_type, self.str_nb_type),
        )
        self.assertEqual(
            as_numba_type(pt.Set[complex]),
            types.Set(self.complex_nb_type),
        )
        self.assertEqual(
            as_numba_type(pt.Tuple[float, float]),
            types.Tuple([self.float_nb_type, self.float_nb_type]),
        )
        self.assertEqual(
            as_numba_type(pt.Tuple[float, complex]),
            types.Tuple([self.float_nb_type, self.complex_nb_type]),
        )

    def test_optional(self):
        self.assertEqual(
            as_numba_type(pt.Optional[float]),
            types.Optional(self.float_nb_type),
        )
        self.assertEqual(
            as_numba_type(pt.Union[str, None]),
            types.Optional(self.str_nb_type),
        )
        self.assertEqual(
            as_numba_type(pt.Union[None, bool]),
            types.Optional(self.bool_nb_type),
        )

        # Optional[x] is a special case of Union[x, None].  We raise a
        # TypingError if the right type is not NoneType.
        with self.assertRaises(TypingError) as raises:
            as_numba_type(pt.Union[int, float])
        self.assertIn("Cannot type Union that is not an Optional",
                      str(raises.exception))

    def test_nested_containers(self):
        IntList = pt.List[int]
        self.assertEqual(
            as_numba_type(pt.List[IntList]),
            types.ListType(types.ListType(self.int_nb_type)),
        )
        self.assertEqual(
            as_numba_type(pt.List[pt.Dict[float, bool]]),
            types.ListType(
                types.DictType(self.float_nb_type, self.bool_nb_type)
            ),
        )
        self.assertEqual(
            as_numba_type(pt.Set[pt.Tuple[pt.Optional[int], float]]),
            types.Set(
                types.Tuple(
                    [types.Optional(self.int_nb_type), self.float_nb_type]
                )
            ),
        )

    def test_annotated(self):
        self.assertEqual(
            as_numba_type(pt_ext.Annotated[int, int]), self.int_nb_type,
        )
        self.assertEqual(
            as_numba_type(pt_ext.Annotated[int, types.int16]), types.int16,
        )
        self.assertEqual(
            as_numba_type(pt_ext.Annotated[np.ndarray, types.float32[:]]),
            types.Array(dtype=types.float32, ndim=1, layout="A"),
        )
        self.assertEqual(
            as_numba_type(pt_ext.Annotated[np.ndarray, types.float32, 1]),
            types.Array(dtype=types.float32, ndim=1, layout="A"),
        )
        self.assertEqual(
            as_numba_type(pt_ext.Annotated[np.ndarray, types.int32, 0]),
            types.Array(dtype=types.int32, ndim=0, layout="A"),
        )
        self.assertEqual(
            as_numba_type(pt_ext.Annotated[np.ndarray, float, 2]),
            types.Array(dtype=self.float_nb_type, ndim=2, layout="A"),
        )
        self.assertEqual(
            as_numba_type(pt_ext.Annotated[np.ndarray, float, 3, "A"]),
            types.Array(dtype=self.float_nb_type, ndim=3, layout="A"),
        )
        self.assertEqual(
            as_numba_type(pt_ext.Annotated[np.ndarray, bool, 2, "F"]),
            types.Array(dtype=self.bool_nb_type, ndim=2, layout="F"),
        )

    def test_jitclass_registers(self):

        @jitclass
        class MyInt:
            x: int

            def __init__(self, value):
                self.x = value

        self.assertEqual(as_numba_type(MyInt), MyInt.class_type.instance_type)

    def test_type_alias(self):
        Pair = pt.Tuple[int, int]
        ListOfPairs = pt.List[Pair]

        pair_nb_type = types.Tuple((self.int_nb_type, self.int_nb_type))
        self.assertEqual(as_numba_type(Pair), pair_nb_type)
        self.assertEqual(
            as_numba_type(ListOfPairs), types.ListType(pair_nb_type)
        )

    def test_overwrite_type(self):
        as_numba_type = AsNumbaTypeRegistry()
        self.assertEqual(as_numba_type(float), self.float_nb_type)
        as_numba_type.register(float, types.float32)
        self.assertEqual(as_numba_type(float), types.float32)
        self.assertNotEqual(as_numba_type(float), self.float_nb_type)

    def test_any_throws(self):
        any_types = [
            pt.Optional[pt.Any],
            pt.List[pt.Any],
            pt.Set[pt.Any],
            pt.Dict[float, pt.Any],
            pt.Dict[pt.Any, float],
            pt.Tuple[int, pt.Any],
        ]

        for bad_py_type in any_types:
            with self.assertRaises(TypingError) as raises:
                as_numba_type(bad_py_type)
            self.assertIn(
                "Cannot infer numba type of python type",
                str(raises.exception),
            )

    def test_bad_union_throws(self):
        bad_unions = [
            pt.Union[str, int],
            pt.Union[int, type(None), pt.Tuple[bool, bool]],
        ]

        for bad_py_type in bad_unions:
            with self.assertRaises(TypingError) as raises:
                as_numba_type(bad_py_type)
            self.assertIn("Cannot type Union", str(raises.exception))

    def test_bad_annotated_throws(self):
        invalid_annotated = [
            pt_ext.Annotated[float, int, 1],
            pt_ext.Annotated[float, bool, "hello", "world"],
            pt_ext.Annotated[np.ndarray, types.int16, 4, "A", "extra"],
        ]
        cannot_infer_type = [
            pt_ext.Annotated[int, "not a type"],
        ]
        value_errors = [
            pt_ext.Annotated[np.ndarray, int, -3],
            pt_ext.Annotated[np.ndarray, types.int16, 4, 1],
        ]

        for bad_annotations, error_type, error_msg in [
            (invalid_annotated, TypingError, "Invalid Annotated syntax."),
            (
                cannot_infer_type,
                TypingError,
                "Cannot infer numba type of python type",
            ),
            (value_errors, ValueError, None),
        ]:
            for bad in bad_annotations:
                with self.assertRaises(error_type) as raises:
                    as_numba_type(bad)
                if error_msg:
                    self.assertIn(error_msg, str(raises.exception))


if __name__ == '__main__':
    unittest.main()
