import time

import numpy as np
from numba import *
from numba.decorators import autojit

@autojit
def slice_array_start(a, start):
    return a[start:]

@autojit
def slice_array_stop(a, stop):
    return a[:stop]

@autojit
def slice_array_step(a, step):
    return a[::step]

@autojit
def slice_array(a, start, stop, step):
    return a[start:stop:step]

@autojit
def time_slicing(a, start, stop, step):
#    with nopython: # should make no difference in timing!
        for i in range(1000000):
            a[start:stop:step] = a[start:stop:step] * a[start:stop:step]

def test_slicing():
    """
    >>> test_slicing()
    """
    a = np.arange(10)
    assert np.all(slice_array(a, 1, 7, 2) == a[1:7:2]) # sanity test

    for start in range(-5, 15):
        assert np.all(slice_array_start(a, start) == a[start:])
        for stop in range(-5, 15):
            assert np.all(slice_array_stop(a, stop) == a[:stop])
            for step in range(-3, 4):
                if step == 0:
                    continue
                assert np.all(slice_array_step(a, step) == a[::step])
                assert np.all(slice_array(a, start, stop, step) ==
                              a[start:stop:step])

def test_slicing_result():
    """
    >>> test_slicing_result()
    array([2, 3, 4, 5, 6, 7, 8, 9])
    """
    a = np.arange(10)
    return slice_array_start(a, 2)

if __name__ == "__main__":
    import numba
    numba.testing.testmod()

#    a = np.arange(10)
#    t = time.time()
#    time_slicing(a, 1, 7, 2)
#    print((time.time() - t))
