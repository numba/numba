import os
import ctypes

from numba import *
from numba import error

@autojit(backend='ast')
def withnopython():
    val = 0.0
    with nopython:
        val += 1.0
        return val

@autojit(backend='ast')
def withnopython_nested(obj):
    result = 0.0
    with nopython:
        with python:
            obj_result = obj.method()
            with nopython:
                result += 1.0

    return obj_result, result

@autojit(backend='ast')
def withnopython_error(obj):
    with nopython:
        return obj.method()

class Class(object):
    def method(self):
        return 20.0

def test_with_no_python():
    assert withnopython() == 1.0
    assert withnopython_nested(Class()) == (20.0, 1.0)
#    try:
#        withnopython_error(Class())
#    except error.NumbaError:
#        pass
#    else:
#        raise Exception("Should have gotten an exception")


if __name__ == "__main__":
    test_with_no_python()