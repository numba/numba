#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
An inplace and an out-of-place implementation of recursive mergesort.
This is not an efficient sort implementation.
The purpose is to demonstrate recursion support.
"""
from __future__ import print_function, division, absolute_import

from timeit import default_timer as timer

import numpy as np

from numba import njit


@njit
def mergesort_inplace(arr):
    "Inplace mergesort"
    assert arr.ndim == 1

    if arr.size > 2:
        mid = arr.size // 2
        first = arr[:mid]
        second = arr[mid:]
        mergesort_inplace(first)
        mergesort_inplace(second)

        left = 0
        right = mid
        while left < mid and right < arr.size:
            if arr[left] <= arr[right]:
                left += 1
            else:
                temp = arr[right]
                right += 1
                # copy left array to the right by one
                for i in range(mid, left, -1):
                    arr[i] = arr[i - 1]
                arr[left] = temp
                left += 1
                mid += 1
    elif arr.size == 2:
        a, b = arr
        arr[0], arr[1] = ((a, b) if a <= b else (b, a))
    return arr


@njit
def mergesort(arr):
    "mergesort"
    assert arr.ndim == 1

    if arr.size > 2:
        mid = arr.size // 2
        first = mergesort(arr[:mid].copy())
        second = mergesort(arr[mid:].copy())

        left = right = 0
        writeidx = 0
        while left < first.size and right < second.size:
            if first[left] <= second[right]:
                arr[writeidx] = first[left]
                left += 1
            else:
                arr[writeidx] = second[right]
                right += 1
            writeidx += 1

        while left < first.size:
            arr[writeidx] = first[left]
            writeidx += 1
            left += 1

        while right < second.size:
            arr[writeidx] = second[right]
            writeidx += 1
            right += 1

    elif arr.size == 2:
        a, b = arr
        arr[0], arr[1] = ((a, b) if a <= b else (b, a))
    return arr


def run(mergesort):
    print(('Running %s' % mergesort.py_func.__name__).center(80, '='))
    # Small case (warmup)
    print("Warmup")
    arr = np.random.random(6)
    expect = arr.copy()
    expect.sort()
    print("unsorted", arr)
    res = mergesort(arr)
    print("  sorted", res)
    # Test correstness
    assert np.all(expect == res)
    print()
    # Large case
    nelem = 10**3
    print("Sorting %d float64" % nelem)
    arr = np.random.random(nelem)
    expect = arr.copy()

    # Run pure python version
    ts = timer()
    mergesort.py_func(arr.copy())
    te = timer()
    print('python took %.3fms' % (1000 * (te - ts)))

    # Run numpy version
    ts = timer()
    expect.sort()
    te = timer()
    print('numpy took %.3fms' % (1000 * (te - ts)))

    # Run numba version
    ts = timer()
    res = mergesort(arr)
    te = timer()
    print('numba took %.3fms' % (1000 * (te - ts)))
    # Test correstness
    assert np.all(expect == res)


def main():
    run(mergesort)
    run(mergesort_inplace)

if __name__ == '__main__':
    main()
