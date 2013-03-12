# -*- coding: utf-8 -*-

"""
Virtual methods using virtual method tables.

Note that for @jit classes, we do not support multiple inheritance with
incompatible base objects. We could use a dynamic offset to base classes,
and adjust object pointers for method calls, like in C++:

    http://www.phpcompiler.org/articles/virtualinheritance.html

However, this is quite complicated, and still doesn't allow dynamic extension
for autojit classes. Instead we will use Dag Sverre Seljebotn's hash-based
virtual method tables:

    https://github.com/numfocus/sep/blob/master/sep200.rst
    https://github.com/numfocus/sep/blob/master/sep201.rst
"""

import numba
import ctypes

from numba.exttypes import compileclass
from numba.typesystem.exttypes import ordering

#------------------------------------------------------------------------
# Static Virtual Method Tables
#------------------------------------------------------------------------

def vtab_name(field_name):
    "Mangle method names for the vtab (ctypes doesn't handle this)"
    if field_name.startswith("__") and field_name.endswith("__"):
        field_name = '__numba_' + field_name.strip("_")
    return field_name

def build_static_vtab(vtab_type, method_pointers):
    """
    Create ctypes virtual method table.

    vtab_type: the vtab struct type (typesystem.struct)
    method_pointers: a list of method pointers ([int])
    """
    assert len(method_pointers) == len(vtab_type.fields)

    vtab_ctype = numba.struct(
        [(vtab_name(field_name), field_type)
            for field_name, field_type in vtab_type.fields]).to_ctypes()

    methods = []
    for (method_name, method_pointer), (field_name, field_type) in zip(
                                        method_pointers, vtab_type.fields):
        assert method_name == field_name
        method_type_p = field_type.to_ctypes()
        cmethod = ctypes.cast(ctypes.c_void_p(method_pointer), method_type_p)
        methods.append(cmethod)

    vtab = vtab_ctype(*methods)
    return vtab

# ______________________________________________________________________
# Build Virtual Method Table

class StaticVTabBuilder(compileclass.VTabBuilder):

    def finalize(self, ext_type):
        ext_type.vtab_type.create_method_ordering(ordering.extending)

    def build_vtab(self, ext_type, method_pointers):
        struct_vtable = numba.struct(
            [(method.name, method.type.pointer())
                 for method in ext_type.vtab_type.methods])

        return build_static_vtab(struct_vtable, method_pointers)

#------------------------------------------------------------------------
# Hash-based virtual method tables
#------------------------------------------------------------------------

def sep201_signature_string(functype):
    return str(functype)

def build_hashing_vtab(vtab_type, method_pointers):
    """
    Build hash-based vtable.
    """
    from extensibletype import methodtable

    n = len(method_pointers)

    ids = [sep201_signature_string(method.type)
               for method in vtab_type.methods]
    flags = [0] * n

    vtab = methodtable.PerfectHashMethodTable(n, ids, flags,
                                              method_pointers)
    return vtab

# ______________________________________________________________________
# Build Hash-based Virtual Method Table

class HashBasedVTabBuilder(compileclass.VTabBuilder):

    def finalize(self, ext_type):
        ext_type.vtab_type.create_method_ordering(ordering.unordered)

    def build_vtab(self, ext_type, method_pointers):
        return build_hashing_vtab(ext_type.vtab_type, method_pointers)
