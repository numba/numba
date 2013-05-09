# -*- coding: utf-8 -*-

"""
Test hash-based virtual method tables.
"""

from __future__ import print_function, division, absolute_import

import itertools

import numba as nb
from numba import *
from numba import typesystem
from numba.exttypes import virtual
from numba.exttypes import methodtable
from numba.exttypes.signatures import Method
from numba.testing.test_support import parametrize, main

#------------------------------------------------------------------------
# Signature enumeration
#------------------------------------------------------------------------

class py_class(object):
    pass

def myfunc1(a):
    pass

def myfunc2(a, b):
    pass

def myfunc3(a, b, c):
    pass

types = list(nb.numeric) + [object_]

array_types = [t[:] for t in types]
array_types += [t[:, :] for t in types]
array_types += [t[:, :, :] for t in types]

all_types = types + array_types

def method(func, name, sig):
    return Method(func, name, sig, False, False)

make_methods1 = lambda: [
    method(myfunc1, 'method', typesystem.function(argtype, [argtype]))
        for argtype in all_types]

make_methods2 = lambda: [
    method(myfunc2, 'method', typesystem.function(argtype1, [argtype1, argtype2]))
        for argtype1, argtype2 in itertools.product(all_types, all_types)]

#------------------------------------------------------------------------
# Table building
#------------------------------------------------------------------------

def make_table(methods):
    table = methodtable.VTabType(py_class, [])
    table.create_method_ordering()

    for i, method in enumerate(methods):
        key = method.name, method.signature.args
        method.lfunc_pointer = i
        table.specialized_methods[key] = method

    assert len(methods) == len(table.specialized_methods)

    return table

def make_hashtable(methods):
    table = make_table(methods)
    hashtable = virtual.build_hashing_vtab(table)
    return hashtable

#------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------

@parametrize(make_methods1(), make_methods2()[:6000])
def test_specializations(methods):
    hashtable = make_hashtable(methods)
    # print(hashtable)

    for i, method in enumerate(methods):
        key = virtual.sep201_signature_string(method.signature, method.name)
        assert hashtable.find_method(key), (i, method, key)


if __name__ == '__main__':
    # test_specializations(make_methods2())
    main()
