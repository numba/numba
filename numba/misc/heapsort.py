"""
Heapsort implementation for Numba arrays.

The algorithm is translated from NumPy 2.5.3.
See numpy/_core/src/npysort/npysort_heapsort.hpp.
"""
from collections import namedtuple

import numpy as np


HeapsortImplementation = namedtuple('HeapsortImplementation', [
    'run_heapsort',
])


def make_heapsort_impl(wrap, lt=None, is_argsort=False):
    kwargs_lite = dict(no_cpython_wrapper=True, _nrt=False)

    if lt is None:
        @wrap(**kwargs_lite)
        def lt(a, b):
            return a < b
    else:
        lt = wrap(**kwargs_lite)(lt)

    if is_argsort:
        @wrap(**kwargs_lite)
        def lessthan(a, b, vals):
            return lt(vals[a], vals[b])
    else:
        @wrap(**kwargs_lite)
        def lessthan(a, b, vals):
            return lt(a, b)

    @wrap(**kwargs_lite)
    def sift_down(arr, start, end, vals, tmp):
        root = start
        while root * 2 + 1 < end:
            child = root * 2 + 1
            if child + 1 < end and lessthan(arr[child], arr[child + 1],
                                            vals):
                child += 1
            if lessthan(tmp, arr[child], vals):
                arr[root] = arr[child]
                root = child
            else:
                break
        arr[root] = tmp

    @wrap(**kwargs_lite)
    def heapsort_inner(arr, vals):
        start = arr.size // 2 - 1
        while start >= 0:
            sift_down(arr, start, arr.size, vals, arr[start])
            start -= 1

        end = arr.size - 1
        while end > 0:
            tmp = arr[end]
            arr[end] = arr[0]
            sift_down(arr, 0, end, vals, tmp)
            end -= 1

    @wrap(no_cpython_wrapper=True)
    def heapsort(arr):
        "Inplace"
        heapsort_inner(arr, None)
        return arr

    @wrap(no_cpython_wrapper=True)
    def argheapsort(arr):
        "Out-of-place"
        idxs = np.arange(arr.size)
        heapsort_inner(idxs, arr)
        return idxs

    return HeapsortImplementation(
        run_heapsort=(argheapsort if is_argsort else heapsort)
    )


def make_jit_heapsort(*args, **kwargs):
    from numba import njit
    return make_heapsort_impl(njit, *args, **kwargs)
