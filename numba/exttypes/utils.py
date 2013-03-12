"Simple utilities related to extension types"

#------------------------------------------------------------------------
# Read state from extension types
#------------------------------------------------------------------------

def get_attributes_type(py_class):
    "Return the attribute struct type of the numba extension type"
    return py_class.__numba_struct_type

def get_vtab_type(py_class):
    "Return the type of the virtual method table of the numba extension type"
    return py_class.__numba_vtab_type

def get_method_pointers(py_class):
    "Return [(method_name, method_pointer)] given a numba extension type"
    return getattr(py_class, '__numba_method_pointers', None)

#------------------------------------------------------------------------
# Type checking
#------------------------------------------------------------------------

def is_numba_class(py_class):
    return hasattr(py_class, '__numba_struct_type')