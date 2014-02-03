import time

import numpy as np
from numba import *
from numba.decorators import autojit

@autojit
def slice_array(a, start, stop, step):
    return a[start:stop:step]

@autojit
def time_slicing(a, start, stop, step):
#    with nopython:
        for i in range(1000000):
            a[start:stop:step]

def test_slicing():
    a = np.arange(10)
    assert np.all(slice_array(a, 1, 7, 2) == a[1:7:2]) # sanity test

    for start in range(-5, 15):
        for stop in range(-5, 15):
            for step in range(-3, 4):
                if step == 0:
                    continue
                assert np.all(slice_array(a, start, stop, step) ==
                              a[start:stop:step])

if __name__ == "__main__":
    test_slicing()

