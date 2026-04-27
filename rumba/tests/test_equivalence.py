"""Verify that Python, Numba, and Rumba produce identical results."""
import math

import numba
import pytest
import rumba


def _equivalent(py_val, numba_val, rumba_val):
    if isinstance(py_val, float):
        return math.isclose(py_val, numba_val) and math.isclose(py_val, rumba_val)
    return py_val == numba_val == rumba_val


def _check(py_func, args):
    numba_func = numba.njit(py_func)
    rumba_func = rumba.njit(py_func)
    py_val = py_func(*args)
    numba_val = numba_func(*args)
    rumba_val = rumba_func(*args)
    assert _equivalent(py_val, numba_val, rumba_val), (
        f"mismatch: Python={py_val!r}, Numba={numba_val!r}, Rumba={rumba_val!r}"
    )


def test_add_integers():
    def add(a, b):
        return a + b

    _check(add, (11, 31))


def test_distance_floats():
    def distance(a, b):
        if a > b:
            return a - b
        return b - a

    _check(distance, (10.5, 2.25))


def test_triangular_loop():
    def triangular(n):
        total = 0
        for i in range(n):
            total += i
        return total

    _check(triangular, (12,))


def test_weighted_sum():
    def weighted_sum(a, b, c):
        return a + b * c

    _check(weighted_sum, (3, 5, 7))
