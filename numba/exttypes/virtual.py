"""
Virtual methods using virtual method tables.
"""

import numba
import ctypes

#------------------------------------------------------------------------
# Virtual Methods
#------------------------------------------------------------------------

def vtab_name(field_name):
    "Mangle method names for the vtab (ctypes doesn't handle this)"
    if field_name.startswith("__") and field_name.endswith("__"):
        field_name = '__numba_' + field_name.strip("_")
    return field_name

def build_vtab(vtab_type, method_pointers):
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

#------------------------------------------------------------------------
# Build Virtual Method Table
#------------------------------------------------------------------------

class StaticVTabBuilder(object):

    def build_vtab_type(self, ext_type):
        "Build vtab type before compiling"
        ext_type.vtab_type = numba.struct(
            [(field_name, field_type.pointer())
                for field_name, field_type in ext_type.methods])

    def build_vtab(self, ext_type, method_pointers):
        return build_vtab(ext_type.vtab_type, method_pointers)