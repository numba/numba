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

from numba.typesystem import *
from numba.typesystem.exttypes import ordering
from extensibletype import methodtable

#------------------------------------------------------------------------
# Virtual Method Table Interface
#------------------------------------------------------------------------

class VTabBuilder(object):
    """
    Build virtual method table for quick calling from Numba.
    """

    def finalize(self, ext_type):
        "Finalize the method table (and fix the order if necessary)"

    def build_vtab(self, ext_type, method_pointers):
        """
        Build a virtual method table.
        The result will be kept alive on the extension type.
        """

#------------------------------------------------------------------------
# Static Virtual Method Tables
#------------------------------------------------------------------------

def vtab_name(field_name):
    "Mangle method names for the vtab (ctypes doesn't handle this)"
    if field_name.startswith("__") and field_name.endswith("__"):
        field_name = '__numba_' + field_name.strip("_")
    return field_name

def build_static_vtab(vtable, vtab_struct):
    """
    Create ctypes virtual method table.

    vtab_type: the vtab struct type (typesystem.struct)
    method_pointers: a list of method pointers ([int])
    """
    vtab_ctype = numba.struct(
        [(vtab_name(field_name), field_type)
            for field_name, field_type in vtab_struct.fields]).to_ctypes()

    methods = []
    for method, (field_name, field_type) in zip(vtable.methods,
                                                vtab_struct.fields):
        method_type_p = field_type.to_ctypes()
        method_void_p = ctypes.c_void_p(method.lfunc_pointer)
        cmethod = ctypes.cast(method_void_p, method_type_p)
        methods.append(cmethod)

    vtab = vtab_ctype(*methods)
    return vtab

# ______________________________________________________________________
# Build Virtual Method Table

class StaticVTabBuilder(VTabBuilder):

    def finalize(self, ext_type):
        ext_type.vtab_type.create_method_ordering(ordering.extending)

    def build_vtab(self, ext_type):
        vtable = ext_type.vtab_type
        return build_static_vtab(vtable, vtable.to_struct())

#------------------------------------------------------------------------
# Hash-based virtual method tables
#------------------------------------------------------------------------

# ______________________________________________________________________
# Type Definitions

# TODO: Use something like CFFI + type conversion to get these
# TODO: types automatically

PyCustomSlots_Entry = numba.struct([
    ('id', char.pointer()),
    ('flags', Py_uintptr_t), # TODO: make flags part of id
    ('ptr', void.pointer()),
])

PyCustomSlots_Table = numba.struct([
    ('flags', uint64),
    ('m_f', uint64),
    ('m_g', uint64),
    ('entries', PyCustomSlots_Entry.pointer()),
    ('n', uint16),
    ('b', uint16),
    ('r', uint8),
    ('reserved', uint8),
    # ('d', uint16[b]), # 'b' trailing displacements
    # ('entries_mem', PyCustomSlot_Entry[n]), # 'n' trailing customslot entries
])

# ______________________________________________________________________
# Hash-table building

sep201_hasher = methodtable.Hasher()

def sep201_signature_string(functype):
    return str(functype)

def hash_signature(functype):
    return sep201_hasher.hash_signature(functype)

def build_hashing_vtab(vtable):
    """
    Build hash-based vtable.
    """
    n = len(vtable.methods)

    ids = [sep201_signature_string(method.type)
               for method in vtable.methods]
    flags = [0] * n

    vtab = methodtable.PerfectHashMethodTable(sep201_hasher)
    vtab.generate_table(n, ids, flags, vtable.method_pointers)

    return vtab

# ______________________________________________________________________
# Build Hash-based Virtual Method Table

class HashBasedVTabBuilder(VTabBuilder):

    def finalize(self, ext_type):
        ext_type.vtab_type.create_method_ordering(ordering.unordered)

    def build_vtab(self, ext_type):
        return build_hashing_vtab(ext_type.vtab_type)
