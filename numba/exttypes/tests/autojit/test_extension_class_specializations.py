# -*- coding: utf-8 -*-

"""
Test properties of specialized classes and test indexing.
"""

from __future__ import print_function, division, absolute_import

from numba import *

@autojit
class C(object):
    def __init__(self, value):
        self.value = value

obj = C(10.0)
print(type(obj).exttype)

specialized_cls = C[{'value': double}]
print(specialized_cls, C, specialized_cls is C)

assert issubclass(specialized_cls, C)
assert isinstance(obj, C)

try:
    C[{'value': int_}]
except KeyError as e:
    assert e.args[0] == {'value': int_}
else:
    raise Exception
