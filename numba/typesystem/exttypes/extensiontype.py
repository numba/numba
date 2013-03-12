# -*- coding: utf-8 -*-

"""
Extension type types.
"""

from numba.typesystem import *

class ExtensionType(NumbaType, minitypes.ObjectType):
    """
    Extension type Numba type.

    Available to users through MyExtensionType.exttype (or
    numba.typeof(MyExtensionType).
    """

    is_extension = True
    is_final = False

    def __init__(self, py_class, **kwds):
        super(ExtensionType, self).__init__(**kwds)
        assert isinstance(py_class, type), ("Must be a new-style class "
                                            "(inherit from 'object')")
        self.name = py_class.__name__
        self.py_class = py_class
        self.symtab = {}  # attr_name -> attr_type
        self.methods = [] # (method_name, func_signature)
        self.methoddict = {} # method_name -> (func_signature, vtab_index)

        self.compute_offsets(py_class)
        self.attribute_struct = None
        self.vtab_type = None

        self.parent_attr_struct = None
        self.parent_vtab_type = None
        self.parent_type = getattr(py_class, "__numba_ext_type", None)

    def compute_offsets(self, py_class):
        from numba.exttypes import extension_types

        self.vtab_offset = extension_types.compute_vtab_offset(py_class)
        self.attr_offset = extension_types.compute_attrs_offset(py_class)

    def set_attributes(self, attribute_list):
        """
        Create the symbol table and attribute struct from a list of
        (varname, attribute_type)
        """
        import numba.symtab

        self.attribute_struct = numba.struct(attribute_list)
        self.symtab.update([(name, numba.symtab.Variable(type))
                               for name, type in attribute_list])

# ______________________________________________________________________
# @jit

class JitExtensionType(ExtensionType):
    "Type for @jit extension types"

    is_jit_extension = True

    def __repr__(self):
        return "<JitExtension %s>" % self.name

    def __str__(self):
        if self.attribute_struct:
            return "<JitExtension %s(%s)>" % (
                self.name, self.attribute_struct.fielddict)
        return repr(self)

# ______________________________________________________________________
# @autojit

class AutojitExtensionType(ExtensionType):
    "Type for @autojit extension types"

    is_autojit_extension = True

    def __repr__(self):
        return "<AutojitExtension %s>" % self.name

    def __str__(self):
        if self.attribute_struct:
            return "<AutojitExtension %s(%s)>" % (
                self.name, self.attribute_struct.fielddict)
        return repr(self)

