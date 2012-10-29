import numpy as np
from numba import *
from numba.decorators import autojit

@autojit(backend='ast')
def slice_array(a, start, stop, step):
    return a[start:stop:step]

def test_slicing():
    a = np.arange(10)
    assert np.all(slice_array(a, 1, 7, 2) == a[1:7:2])

if __name__ == "__main__":
    test_slicing()