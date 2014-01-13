# -*- coding: utf-8 -*-

"""
Extension type types.
"""

from functools import partial

from numba.traits import traits, Delegate
from numba.typesystem import NumbaType

@traits
class ExtensionType(NumbaType):
    """
    Extension type Numba type.

    Available to users through MyExtensionType.exttype (or
    numba.typeof(MyExtensionType).
    """

    typename = "extension"
    argnames = ["py_class"]
    flags = ["object"]
    is_final = False

    methoddict = Delegate('vtab_type')
    untyped_methods = Delegate('vtab_type')
    specialized_methods = Delegate('vtab_type')
    methodnames = Delegate('vtab_type')
    add_method = Delegate('vtab_type')

    attributedict = Delegate('attribute_table')
    attributes = Delegate('attribute_table')

    def __init__(self, py_class):
        super(ExtensionType, self).__init__(py_class)
        assert isinstance(py_class, type), ("Must be a new-style class "
                                            "(inherit from 'object')")
        self.name = py_class.__name__
        self.py_class = py_class
        self.extclass = None

        self.symtab = {}  # attr_name -> attr_type

        self.compute_offsets(py_class)

        self.attribute_table = None
        self.vtab_type = None

        self.parent_attr_struct = None
        self.parent_vtab_type = None
        self.parent_type = getattr(py_class, "__numba_ext_type", None)

    def compute_offsets(self, py_class):
        from numba.exttypes import extension_types

        self.vtab_offset = extension_types.compute_vtab_offset(py_class)
        self.attr_offset = extension_types.compute_attrs_offset(py_class)

# ______________________________________________________________________
# @jit

class jit_exttype(ExtensionType):
    "Type for @jit extension types"

    def __repr__(self):
        return "<JitExtension %s>" % self.name

    def __str__(self):
        if self.attribute_table:
            return "<JitExtension %s(%s)>" % (
                    self.name, self.attribute_table.strtable())
        return repr(self)

# ______________________________________________________________________
# @autojit

class autojit_exttype(ExtensionType):
    "Type for @autojit extension types"

    def __repr__(self):
        return "<AutojitExtension %s>" % self.name

    def __str__(self):
        if self.attribute_table:
            return "<AutojitExtension %s(%s)>" % (
                    self.name, self.attribute_table.strtable())
        return repr(self)
