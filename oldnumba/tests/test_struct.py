import os

from numba import *
from numba import error

import numpy as np

#------------------------------------------------------------------------
# Structs as locals
#------------------------------------------------------------------------

struct_type = struct_([('a', char.pointer()), ('b', int_)])

@autojit(backend='ast', locals=dict(value=struct_type))
def struct_local():
    value.a = "foo"
    value.b = 10
    return value.a, value.b

@autojit(backend='ast', locals=dict(value=struct_type))
def struct_local_inplace():
    value.a = "foo"
    value.b = 10
    value.b += 10.0
    return value.a, value.b

# TODO: structs from objects
#@autojit
#def struct_as_arg(arg):
#    arg.a = "foo"
#    return arg.a
#
#@autojit(backend='ast', locals=dict(value=struct_type))
#def call_struct_as_arg():
#    return struct_as_arg(value)

@autojit(backend='ast', locals=dict(value=struct_type))
def struct_local_copy():
    value.a = "foo"
    value.b = 10
    value2 = value
    return value2.a, value2.b

def test_struct_locals():
    result = struct_local()
    assert result == ("foo", 10), result

    result = struct_local_inplace()
    assert result == ("foo", 20), result

#    result = call_struct_as_arg()
#    assert result == "foo", result

    result = struct_local_copy()
    assert result == ("foo", 10), result

#------------------------------------------------------------------------
# Struct indexing
#------------------------------------------------------------------------

@autojit(backend='ast', locals=dict(value=struct_type))
def struct_indexing_strings():
    value['a'] = "foo"
    value['b'] = 10
    return value['a'], value['b']

@autojit(backend='ast', locals=dict(value=struct_type))
def struct_indexing_ints():
    value[0] = "foo"
    value[1] = 10
    return value[0], value[1]

def test_struct_indexing():
    assert struct_indexing_strings() == ("foo", 10)
    assert struct_indexing_ints() == ("foo", 10)

#------------------------------------------------------------------------
# Record arrays
#------------------------------------------------------------------------

@autojit(backend='ast')
def record_array(array):
    array[0].a = 4
    array[0].b = 5.0

def test_record_array():
    struct_type = struct_([('a', int32), ('b', double)])
    struct_dtype = struct_type.get_dtype()

    array = np.empty((1,), dtype=struct_dtype)
    record_array(array)
    assert array[0]['a'] == 4, array[0]
    assert array[0]['b'] == 5.0, array[0]

#------------------------------------------------------------------------
# Object Coercion
#------------------------------------------------------------------------

struct_type = struct_([('a', int_), ('b', double)])

@autojit(backend='ast', locals=dict(value=struct_type))
def coerce_to_obj():
    value.a = 10
    value.b = 20.2
    return object_(value)

def test_coerce_to_obj():
    print((coerce_to_obj()))

if __name__ == "__main__":
    print((struct_local_copy()))
    # print call_struct_as_arg()
    test_struct_locals()
    test_record_array()
    test_coerce_to_obj()
    test_struct_indexing()
