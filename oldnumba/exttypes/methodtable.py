# -*- coding: utf-8 -*-

"""
Virtual method table types and ordering.
"""

from __future__ import print_function, division, absolute_import

import numba
from numba.exttypes import ordering
from numba.exttypes.types import methods

#------------------------------------------------------------------------
# Virtual Method Table Type
#------------------------------------------------------------------------

class VTabType(object):
    """
    Virtual method table type.
    """

    def __init__(self, py_class, parents):
        self.py_class = py_class

        # List of parent vtable types
        self.parents = parents

        # method_name -> Method
        self.methoddict = {}

        # method_name -> Method
        self.untyped_methods = {}

        # specialized methods (i.e. autojit method specializations)
        # (method_name, method_argtypes) -> Method
        self.specialized_methods = {}

        # Set once create_method_ordering is called,
        # list of ordered method names
        self.methodnames = None

        # Set of inherited method ({ method_name })
        self.inherited = set()

    def create_method_ordering(self, orderer=ordering.unordered):
        """
        Create a consistent method ordering with the base types.

            ordering ∈ { unordered, extending, ... }
        """
        self.methodnames = orderer(ordering.VTable(self))

    def add_method(self, method):
        """
        Add a method to the vtab type and verify it with any parent
        method signatures.
        """
        if method.name in self.methoddict:
            # Patch current signature after type inference
            signature = self.get_signature(method.name)
            assert methods.equal_signature_args(method.signature, signature)
            if signature.return_type is None:
                signature.return_type = method.signature.return_type
            else:
                assert signature.return_type == method.signature.return_type, \
                                                            method.signature

        self.methoddict[method.name] = method

    def get_signature(self, method_name):
        "Get the signature for the given method name. Returns ExtMethodType"
        method = self.methoddict[method_name]
        return method.signature

    def to_struct(self):
        return numba.struct([(m.name, m.signature.pointer())
                                 for m in self.methods])

    @property
    def methods(self):
        "Return methods in the order they were set in"
        assert self.methodnames is not None
        methods = map(self.methoddict.get, self.methodnames)
        return list(methods) + self.specialized_methods.values()

    @property
    def llvm_methods(self):
        for m in self.methods:
            yield m.lfunc

    @property
    def method_pointers(self):
        for m in self.methods:
            yield m.lfunc_pointer

    @classmethod
    def empty(cls, py_class):
        "Create an empty finalized vtable type"
        vtable = cls(py_class, [])
        vtable.create_method_ordering()
        return vtable
