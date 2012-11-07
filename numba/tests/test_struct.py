import os

from numba import *
from numba import error

import numpy as np

@autojit(backend='ast', locals=dict(value=struct(a=char.pointer(), b=int_)))
def struct_local():
    value.a = "foo"
    value.b = 10
    return value.a, value.b

@autojit(backend='ast', locals=dict(value=struct(a=char.pointer(), b=int_)))
def struct_local_inplace():
    value.a = "foo"
    value.b = 10
    value.b += 10.0
    return value.a, value.b

def test_struct_locals():
    result = struct_local()
    assert result == ("foo", 10), result
    result = struct_local_inplace()
    assert result == ("foo", 20), result

@autojit(backend='ast', locals=dict(value=struct(a=char.pointer(), b=int_)))
def struct_indexing_strings():
    value['a'] = "foo"
    value['b'] = 10
    return value['a'], value['b']

@autojit(backend='ast', locals=dict(value=struct(a=char.pointer(), b=int_)))
def struct_indexing_ints():
    value[0] = "foo"
    value[1] = 10
    return value[0], value[1]

def test_struct_indexing():
    assert struct_indexing_strings() == ("foo", 10)
    assert struct_indexing_ints() == ("foo", 10)

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
    test_struct_indexing()
