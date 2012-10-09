import os

from numba import *
from numba import error

@autojit(backend='ast', locals=dict(value=double))
def locals_override(obj):
    value = obj.method()
    with nopython:
        return value * value

class Class(object):
    def method(self):
        return 20.0

def test_locals_override():
    assert locals_override(Class()) == 400.0

if __name__ == "__main__":
    test_locals_override()