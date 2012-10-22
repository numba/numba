cimport cython
from cpython cimport PyObject

import sys
import ctypes
import doctest

from numba import *

ctypedef object (*tp_new_func)(PyObject *, PyObject *, PyObject *)

cdef extern from *:
    ctypedef unsigned long Py_uintptr_t

    ctypedef struct PyTypeObject:
        tp_new_func tp_new
        long tp_dictoffset
        Py_ssize_t tp_itemsize
        Py_ssize_t tp_basicsize


cdef Py_uintptr_t align(Py_uintptr_t p, size_t alignment) nogil:
    cdef size_t offset

    with cython.cdivision(True):
        offset = p % alignment

    if offset > 0:
        p += alignment - offset

    return p

cdef void *align_pointer(void *memory, size_t alignment) nogil:
    "Align pointer memory on a given boundary"
    return <void *> align(<Py_uintptr_t> memory, alignment)

def compute_vtab_offset(py_class):
    cdef PyTypeObject *type_p = <PyTypeObject *> py_class
    return align(type_p.tp_basicsize, 8)

def compute_attrs_offset(py_class):
    return align(compute_vtab_offset(py_class) + sizeof(void *), 8)

def create_new_extension_type(name, bases, dict, struct_type, vtab_type,
                              llvm_methods, method_pointers):
    cdef PyTypeObject *ext_type_p
    cdef Py_ssize_t vtab_offset

    orig_new = dict.get('__new__', None)
    def new(cls, *args, **kwds):
        cdef PyObject *obj_p
        cdef void **vtab_location

        if orig_new is not None:
            obj = orig_new(cls, *args, **kwds)
        else:
            obj = super(cls, ext_type).__new__(cls, *args, **kwds)

        if (cls.__numba_vtab is not ext_type.__numba_vtab or not
                isinstance(obj, cls)):
            # Subclass will set the vtab and attributes
            return obj

        # It is our responsibility to set the vtab and the ctypes attributes
        # Other fields are 0/NULL
        obj_p = <PyObject *> obj

        vtab_location = <void **> ((<char *> obj_p) + vtab_offset)
        vtab_location[0] = <void *> <Py_uintptr_t> cls.__numba_vtab_p

        attrs_pointer = (<Py_uintptr_t> obj_p) + vtab_offset + sizeof(void *)
        obj._numba_attrs = ctypes.cast(attrs_pointer,
                                       cls.__numba_struct_ctype_p)[0]

        return obj

    dict['__new__'] = staticmethod(new)
    ext_type = type(name, bases, dict)
    assert isinstance(ext_type, type)

    ext_type_p = <PyTypeObject *> ext_type

    # Object offset for vtab is lower
    # Object attributes are located at lower + sizeof(void *), and end at
    # upper
    struct_ctype = struct_type.to_ctypes()
    vtab_offset = compute_vtab_offset(ext_type)
    attrs_offset = compute_attrs_offset(ext_type)
    upper = attrs_offset + ctypes.sizeof(struct_ctype)
    upper = align(upper, 8)

    # print 'basicsize/vtab_offset/upper', ext_type_p.tp_basicsize, lower, upper
    ext_type_p.tp_basicsize = upper
    if ext_type_p.tp_itemsize:
        raise NotImplementedError("Subclassing variable sized objects")

    ext_type.__numba_vtab_offset = vtab_offset
    ext_type.__numba_obj_end = upper
    ext_type.__numba_attr_offset = attrs_offset

    vtab_ctype = vtab_type.to_ctypes()
    methods = []
    for method_pointer, (field_name, field_type) in zip(method_pointers,
                                                        vtab_ctype._fields_):
        cmethod = field_type(method_pointer)
        methods.append(cmethod)

    vtab = vtab_ctype(*methods)
    ext_type.__numba_vtab = vtab
    vtab_p = ctypes.byref(ext_type.__numba_vtab)

    ext_type.__numba_vtab_p = ctypes.cast(vtab_p, ctypes.c_void_p).value
    ext_type.__numba_orig_tp_new = <Py_uintptr_t> ext_type_p.tp_new
    ext_type.__numba_struct_type = struct_type
    ext_type.__numba_struct_ctype_p = struct_type.pointer().to_ctypes()
    ext_type.__numba_lfuncs = llvm_methods

    return ext_type
