import numpy as np
import itertools
from numba import njit, config
from numba.tests.support import TestCase


class TestTypes(TestCase):

    def setUp(self) -> None:
        if config.USE_LEGACY_TYPE_SYSTEM:
            self.skipTest("This test is only for the new type system")
        return super().setUp()

    def test_return_types(self):
        @njit
        def foo(x):
            return x

        cases = [
            # Python types
            1,
            1.2,
            (1 + 2j),
            True,
            # NumPy types
            np.int32(1),
            np.float64(1.2),
            np.complex64(1 + 2j),
            np.complex128(1 + 2j),
            np.bool_(True),
            np.datetime64('2020-01-01'),
            np.timedelta64(1, 'D'),
        ]

        for case in cases:
            self.assertEqual(foo(case), case)
            self.assertEqual(type(foo(case)), type(case))


class TestDunderMethods(TestCase):

    type_cases = [
        True,
        10,
        1.1,
        1 + 2j,
        np.bool_(True),
        np.int8(1),
        np.int16(2),
        np.int32(3),
        np.int64(4),
        np.uint8(5),
        np.uint16(6),
        np.uint32(7),
        np.uint64(8),
        np.float16(1.1),
        np.float32(3.2),
        np.float64(5.5),
        # np.complex64((20+5j)),
        np.complex128((4 + 3j))
    ]

    def setUp(self) -> None:
        if config.USE_LEGACY_TYPE_SYSTEM:
            self.skipTest("This test is only for the new type system")
        return super().setUp()

    def test_dunder_add(self):
        @njit
        def foo(a, b):
            return a.__add__(b)

        for x, y in itertools.product(self.type_cases, self.type_cases):
            res = foo(x, y)
            py_res = foo.py_func(x, y)

            assert res == py_res, (
                f"Failed for {x} and {y};" +
                f" gave answer {res} should be {py_res}")
            assert type(res) is type(py_res), (
                f"Failed for type {type(x)} and {type(y)};" +
                f" gave answer {type(res)} should be {type(py_res)}")

    def test_dunder_radd(self):
        @njit
        def foo(a, b):
            return a.__radd__(b)

        for x, y in itertools.product(self.type_cases, self.type_cases):
            res = foo(x, y)
            py_res = foo.py_func(x, y)

            assert res == py_res, (
                f"Failed for {x} and {y};" +
                f" gave answer {res} should be {py_res}")
            assert type(res) is type(py_res), (
                f"Failed for type {type(x)} and {type(y)};" +
                f" gave answer {type(res)} should be {type(py_res)}")

    def test_add(self):
        @njit
        def foo(a, b):
            return a + b

        for x, y in itertools.product(self.type_cases, self.type_cases):
            res = foo(x, y)
            py_res = foo.py_func(x, y)

            assert res == py_res, (
                f"Failed for {x} and {y};" +
                f" gave answer {res} should be {py_res}")
            assert type(res) is type(py_res), (
                f"Failed for type {type(x)} and {type(y)};" +
                f" gave answer {type(res)} should be {type(py_res)}")
