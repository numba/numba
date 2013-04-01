"Simple utilities related to extension types"

#------------------------------------------------------------------------
# Type checking
#------------------------------------------------------------------------

def is_numba_class(py_class):
    return hasattr(py_class, '__numba_ext_type')

def get_all_numba_bases(py_class):
    seen = set()

    bases = []
    for base in py_class.__mro__[::-1]:
        if is_numba_class(base) and base.exttype not in seen:
            seen.add(base.exttype)
            bases.append(base)

    return bases[::-1]

def get_numba_bases(py_class):
    return list(filter(is_numba_class, py_class.__bases__))

def get_class_dict(unspecialized_autojit_py_class):
    return unspecialized_autojit_py_class.__numba_class_dict