# -*- coding: utf-8 -*-

"""
Virtual method table types and ordering.
"""

from numba import error
from numba.typesystem import *
from numba.typesystem.exttypes import ordering

#------------------------------------------------------------------------
# Virtual Method Table Type
#------------------------------------------------------------------------

class VTabType(NumbaType):
    """
    Virtual method table type.
    """

    def __init__(self, py_class, parents):
        self.py_class = py_class

        # List of parent vtable types
        self.parents = parents

        # method_name -> Method
        self.methoddict = {}

        # Set once create_method_ordering is called,
        # list of ordered method names
        self.methodnames = None

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
            assert method.signature.args == signature.args
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
