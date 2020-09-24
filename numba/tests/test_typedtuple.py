import numpy as np

from numba import int32, float64, njit
from numba.typed import typed_tuple
from numba.typed import List as NumbaList
from numba.tests.support import TestCase, MemoryLeakMixin

from collections import namedtuple
import typing as py_typing


class Point1(py_typing.NamedTuple):
    x: float
    y: int


Point2 = namedtuple("Point2", ("x", "y"))


class PointedArray(py_typing.NamedTuple):
    x: np.ndarray
    idx: int


class PointedList(py_typing.NamedTuple):
    x: py_typing.List[float]
    idx: int


class PointPair(py_typing.NamedTuple):
    p1: Point1
    p2: Point1


class TestTypedTuple(TestCase, MemoryLeakMixin):

    def test_py_constructor(self):
        typed_tuple(Point1)
        typed_tuple(Point2, dict(x=float64, y=int32))

        for P in (Point1, Point2):
            self.assertTrue(hasattr(P, "_numba_type_"))

            p_int = P(1, 1)
            self.assertIsInstance(p_int.x, float)
            self.assertIsInstance(p_int.y, int)

            p_float = P(3.14, 3.14)
            self.assertIsInstance(p_float.x, float)
            self.assertIsInstance(p_float.y, int)

    def test_with_spec(self):
        typed_tuple(PointedArray, dict(x=int32[:]))

        @njit
        def get_value(p_list):
            return p_list.x[p_list.idx]

        p = PointedArray(np.array([10, 11, 12, 13], dtype=np.int32), 1)
        self.assertEqual(get_value(p), 11)

    def test_jit_constructor(self):
        typed_tuple(PointedList)

        @njit
        def get_value(p):
            return p.x[p.idx]

        @njit
        def get_max_plist(items):
            x = NumbaList()
            idx = 0

            for i, value in enumerate(items):
                value = float(value)
                x.append(value)

                if x[idx] < value:
                    idx = i

            return PointedList(x, idx)

        l = NumbaList([0., 1., 2., 3., 4.])

        p1 = PointedList(l, 4)
        self.assertPreciseEqual(get_value(p1), 4.)

        p2 = get_max_plist([0, 1, 2, 3, 4])
        self.assertEqual(p1, p2)

    def test_nested(self):
        typed_tuple(Point1)
        typed_tuple(PointPair)

        @njit
        def get_delta(pair):
            return Point1(pair.p2.x - pair.p1.x, pair.p2.y - pair.p1.y)

        p = PointPair(Point1(0, 1), Point1(4, 4))
        self.assertEqual(get_delta(p), Point1(4, 3))

    def test_repeat_calls(self):
        typed_tuple(typed_tuple(Point1))

        p = Point1(2, 3)
        self.assertPreciseEqual(p.x, 2.)
        self.assertPreciseEqual(p.y, 3)
