import os

from numba import *
from numba import error

import numpy as np

@autojit(backend='ast', locals=dict(value=struct(a=char.pointer(), b=int_)))
def struct_local():
    value.a = "foo"
    value.b = 10
    return value.a, value.b

def test_struct_locals():
    print struct_local()
    assert struct_local() == ("foo", 10)

# ----------------

@autojit(backend='ast')
def record_array(array):
    array[0].a = 4
    array[0].b = 5.0

def test_record_array():
    struct_type = struct([('a', int32), ('b', double)])
    struct_dtype = struct_type.get_dtype()

    array = np.empty((1,), dtype=struct_dtype)
    record_array(array)
    assert array[0]['a'] == 4, array[0]
    assert array[0]['b'] == 5.0, array[0]

# ----------------

struct_type = struct([('a', int_), ('b', double)])

@autojit(backend='ast', locals=dict(value=struct_type))
def coerce_to_obj():
    value.a = 10
    value.b = 20.2
    return object_(value)

def test_coerce_to_obj():
    print coerce_to_obj()

if __name__ == "__main__":
    test_struct_locals()
    test_record_array()
    test_coerce_to_obj()
