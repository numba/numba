cimport cython
from cpython cimport PyObject

import ctypes

ctypedef object (*tp_new_func)(PyObject *, PyObject *, PyObject *)

cdef extern from *:
    ctypedef unsigned long Py_uintptr_t

    ctypedef struct PyTypeObject:
        tp_new_func tp_new
        long tp_dictoffset
        Py_ssize_t tp_itemsize
        Py_ssize_t tp_basicsize


cdef Py_uintptr_t align(Py_uintptr_t memory, size_t alignment) nogil:
    cdef Py_uintptr_t aligned_p = <Py_uintptr_t> memory
    cdef size_t offset

    with cython.cdivision(True):
        offset = aligned_p % alignment

    if offset > 0:
        aligned_p += alignment - offset

cdef void *align_pointer(void *memory, size_t alignment) nogil:
    "Align pointer memory on a given boundary"
    return <void *> align(<Py_uintptr_t> memory, alignment)

cdef object tp_new(PyObject *subtype, PyObject *args, PyObject *kwds):
    cdef tp_new_func orig_tp_new
    cdef PyObject *obj_p
    cdef Py_ssize_t vtab_offset
    cdef void **vtab_location

    subtype_obj = <object> subtype
    vtab_offset = subtype_obj.__numba_vtab_offset
    orig_tp_new = <tp_new_func> <Py_uintptr_t> subtype_obj.__numba_orig_tp_new

    obj = orig_tp_new(subtype, args, kwds)
    obj_p = <PyObject *> obj_p

    if isinstance(obj, subtype_obj):
        # Initialize vtab, other fields are 0/NULL
        vtab_location = <void **> ((<char *> obj_p) + vtab_offset)
        vtab_location[0] = <void *> <Py_uintptr_t> subtype_obj.__numba_vtab_p

    return obj

def create_new_extension_type(name, bases, dict, struct_type, vtab_type,
                              llvm_methods, method_pointers, wrapper_methods):
    """

    """
    ext_type = type(name, bases, dict)
    assert isinstance(ext_type, type)

    cdef PyTypeObject *ext_type_p = <PyTypeObject *> ext_type

    # Object offset for vtab is lower
    # Object attributes are located at lower + sizeof(void *), and end at
    # upper
    struct_ctype = struct_type.to_ctypes()
    lower = align(ext_type.tp_basicsize, 8)
    upper = (lower + ctypes.sizeof(ctypes.c_void_p) +
             ctypes.sizeof(struct_ctype))
    upper = align(upper, 8)

    ext_type_p.tp_basicsize = upper
    if ext_type.tp_itemsize:
        raise NotImplementedError("Subclassing variable sized objects")

    ext_type.__numba_vtab_offset = lower
    ext_type.__numba_obj_end = upper

    vtab_ctype = vtab_type.to_ctypes()
    ext_type.__numba_vtab = vtab_ctype(*method_pointers)
    vtab_p = ctypes.byref(ext_type.__numba_vtab)
    ext_type.__numba_vtab_p = ctypes.cast(vtab_p, ctypes.c_void_p).value
    ext_type.__numba_orig_tp_new = <Py_uintptr_t> ext_type_p.tp_new

    return ext_type
