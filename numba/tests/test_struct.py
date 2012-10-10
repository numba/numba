import os

from numba import *
from numba import error

@autojit(backend='ast', locals=dict(value=struct(a=char.pointer(), b=int_)))
def struct_local():
    value.a = "foo"
    value.b = 10
    return value.a, value.b

def test_struct_locals():
    print struct_local()
    assert struct_local() == ("foo", 10)

if __name__ == "__main__":
    test_struct_locals()