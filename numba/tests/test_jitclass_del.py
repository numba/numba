import numpy as np
import gc

from numba import njit, types
from numba.experimental import jitclass
from numba.tests.support import TestCase


spec = [
    ("arr", types.int64[::1]),
]


@jitclass(spec)
class WithDel:
    def __init__(self, arr):
        self.arr = arr

    def __del__(self):
        self.arr[0] += 1


@njit
def make_one(arr):
    WithDel(arr)


@njit
def make_many(arr, n):
    for _ in range(n):
        WithDel(arr)


@njit
def alias_refs(arr):
    inst = WithDel(arr)
    other = inst
    # Both names refer to the same instance; destructor should still run once.
    return


@njit
def return_instance(arr):
    return WithDel(arr)


spec_raise = [("arr", types.int64[::1])]


@jitclass(spec_raise)
class WithDelRaises:
    def __init__(self, arr):
        self.arr = arr

    def __del__(self):
        # Intentional error: should be swallowed.
        _ = 1 // 0


@njit
def make_raises(arr):
    WithDelRaises(arr)


class TestJitClassDel(TestCase):
    def test_del_called(self):
        arr = np.zeros(1, dtype=np.int64)
        make_one(arr)
        self.assertEqual(arr[0], 1)

    def test_del_multiple_instances(self):
        arr = np.zeros(1, dtype=np.int64)
        make_many(arr, 5)
        self.assertEqual(arr[0], 5)

    def test_del_alias_single_call(self):
        arr = np.zeros(1, dtype=np.int64)
        alias_refs(arr)
        self.assertEqual(arr[0], 1)

    def test_del_from_python_box_lifetime(self):
        arr = np.zeros(1, dtype=np.int64)
        inst = WithDel(arr)
        # Drop the Python box and force collection; meminfo destructor should run.
        del inst
        gc.collect()
        self.assertEqual(arr[0], 1)

    def test_del_errors_are_suppressed(self):
        arr = np.zeros(1, dtype=np.int64)
        make_raises(arr)
        # The destructor error is swallowed; no exception propagated.
        self.assertEqual(arr[0], 0)

    def test_del_on_returned_value(self):
        arr = np.zeros(1, dtype=np.int64)
        inst = return_instance(arr)
        del inst
        gc.collect()
        self.assertEqual(arr[0], 1)
