# -*- coding: utf-8 -*-

"""
Virtual method table types and ordering.
"""

from numba import error
from numba.typesystem import *

#------------------------------------------------------------------------
# Virtual Method Ordering
#------------------------------------------------------------------------

def unordered(parent_vtables, methoddict):
    return methoddict.itervalues()

def extending(parent_vtables, methoddict):
    """
    Order the virtual methods according to the given parent vtables, i.e.
    we can only extend existing vtables.
    """
    if not parent_vtables:
        return unordered(parent_vtables, methoddict)

    parents = sorted(parent_vtables, key=lambda vtab: len(vtab.methoddict))
    biggest_vtab = parents[-1]

    appending_methods = set(methoddict) - set(biggest_vtab.methodnames)
    return biggest_vtab.methodnames + list(appending_methods)

# ______________________________________________________________________
# Validate Virtual Method Order

def validate_vtab_compatibility(parent_vtables, vtab):
    parents = sorted(parent_vtables, key=lambda vtab: len(vtab.methoddict))
    vtabs = parents + [vtab]

    for vtab_smaller, vtab_bigger in zip(vtabs, vtabs[1:]):
        names1 = vtab_smaller.methodnames
        names2 = vtab_bigger.methodnames[len(vtab_smaller.methodnames)]

        if names1 != names2:
            raise error.NumbaError(
                "Cannot create compatible method ordering for "
                "base classes '%s' and '%s'" % (
                                            vtab_smaller.py_class.__name__,
                                            vtab_bigger.py_class.__name__))

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

    def create_method_ordering(self, ordering=unordered):
        """
        Create a consistent method ordering with the base types.

            ordering ∈ { unordered, extending, ... }
        """
        self.methodnames = ordering(self.parents, self.methoddict)

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
