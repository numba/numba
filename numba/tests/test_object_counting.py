import sys
import ctypes

from numba import *
import numpy as np

class Unique(object):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return "Unique(%d)" % self.value

@autojit(backend='ast')
def use_objects(obj_array):
    for i in range(10):
        var = obj_array[i]
        print var

def test_refcounting():
    import test_support

    L = np.array([Unique(i) for i in range(10)], dtype=np.object)
    assert all(sys.getrefcount(obj) == 3 for obj in L)
    with test_support.StdoutReplacer() as out:
        use_objects(L)

    expected = "\n".join("Unique(%d)" % i for i in range(10)) + '\n'
    assert out.getvalue() == expected
    assert all(sys.getrefcount(obj) == 3 for obj in L)

if __name__ == "__main__":
    test_refcounting()