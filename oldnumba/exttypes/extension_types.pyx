"""
This modules creates extension types.

See also numba.extension_type_inference
"""

cimport cython
from numba._numba cimport *

import sys
import ctypes

import numba

ctypedef object (*tp_new_func)(PyObject *, PyObject *, PyObject *)
ctypedef void (*destructor)(PyObject *)
ctypedef int (*visitproc)(PyObject *, void *)

cdef extern from *:
    ctypedef struct PyTypeObject:
        tp_new_func tp_new
        destructor tp_dealloc
        int (*tp_traverse)(PyObject *o, visitproc v, void *a)
        int (*tp_clear)(PyObject *o)

        long tp_dictoffset
        Py_ssize_t tp_itemsize
        Py_ssize_t tp_basicsize
        PyTypeObject *tp_base

#------------------------------------------------------------------------
# Code to compute table offsets in the objects
#------------------------------------------------------------------------

cdef Py_uintptr_t align(Py_uintptr_t p, size_t alignment) nogil:
    "Align on a boundary"
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
    "Returns the vtab pointer offset in the object"
    offset = getattr(py_class, '__numba_vtab_offset', None)
    if offset:
        return offset

    cdef PyTypeObject *type_p = <PyTypeObject *> py_class
    return align(type_p.tp_basicsize, 8)

def compute_attrs_offset(py_class):
    "Returns the start of the attribute struct"
    offset = getattr(py_class, '__numba_attr_offset', None)
    if offset:
        return offset

    return align(compute_vtab_offset(py_class) + sizeof(void *), 8)

#------------------------------------------------------------------------
# Code to handle tp_dealloc, tp_traverse and tp_clear
#------------------------------------------------------------------------
# Currently we don't use this, since it seems to bother the ctypes struct
# in the object dict, but only if we access obj._numba_attrs._fields_
# We will want to generate descriptors to access attributes from Python
# directly instead of ctypes, in which case we need to set these functions
# on the extension type object.

def compute_object_offsets(ext_type, struct_offset):
    """
    Given an typesystem.ExtensionType, compute the offsets of object
    pointers, relative to the start of the attribute offsets.
    """
    offsets = []
    attr_struct = ext_type.attribute_table.to_struct()
    for field_name, field_type in attr_struct.fields:
        if field_type.is_object or field_type.is_array:
            offset = attr_struct.offsetof(field_name)
            offsets.append(struct_offset + offset)

    return offsets

cdef inline list getoffsets(type_obj):
    cdef list offsets = type_obj.__numba_object_offset
    return offsets

cdef inline void setnone(PyObject **slot):
    cdef PyObject *tmp

    tmp = slot[0]
    Py_INCREF(<PyObject *> None)
    slot[0] = <PyObject *> None
    Py_XDECREF(tmp)

cdef void tp_dealloc(PyObject *self):
    """
    Dealloc for numba extension types.

    This is set on the base numba extension type, since the offsets include
    the offsets of all superclasses. I.e. an instance of a subtype will use
    the deallocator of the supertype, which uses the offsets of the subtype.
    """
    type_obj = <object> self.ob_type
    cdef PyTypeObject *ob_type = <PyTypeObject *> type_obj
    cdef Py_ssize_t offset
    cdef PyObject **slot

    for offset in getoffsets(type_obj):
        slot = <PyObject **> ((<char *> <PyObject *> self) + offset)
        # Py_CLEAR(slot[0])
        setnone(slot) # Does ctypes collect and traverse for us?

    # Call base class dealloc
    if ob_type.tp_base.tp_dealloc != NULL:
        ob_type.tp_base.tp_dealloc(<PyObject *> self)

cdef int tp_clear(PyObject *self):
    """
    tp_clear for numba extension types.
    """
    cdef PyTypeObject *ob_type = <PyTypeObject *> self.ob_type

    # Let superclass clear its attributes
    if ob_type.tp_base.tp_clear != NULL:
        ob_type.tp_base.tp_clear(self)

    # Clear our own attributes
    cdef Py_ssize_t offset
    cdef PyObject **slot


    for offset in getoffsets(<object> ob_type):
        slot = <PyObject **> ((<char *> <PyObject *> self) + offset)
        setnone(slot)

    return 0

cdef int tp_traverse(PyObject *self, visitproc visit, void *arg):
    """
    tp_traverse for numba extension types.
    """
    cdef PyTypeObject *ob_type = <PyTypeObject *> self.ob_type
    cdef int result

    if ob_type.tp_base.tp_traverse != NULL:
        result = ob_type.tp_base.tp_traverse(self, visit, arg)
        if result:
            return result

    cdef Py_ssize_t offset
    cdef PyObject *obj

    for offset in getoffsets(<object> ob_type):
        obj = (<PyObject **> ((<char *> <PyObject *> self) + offset))[0]
        if obj != NULL:
            result = visit(obj, arg)
            if result:
                return result

    return 0

#------------------------------------------------------------------------
# Simple Vtable Wrappers
#------------------------------------------------------------------------

class StaticVtableWrapper(object):
    def __init__(self, vtable):
        self.vtable = vtable

    def get_vtable_pointer(self):
        vtab_p = ctypes.byref(self.vtable)
        return ctypes.cast(vtab_p, ctypes.c_void_p).value

cdef class DynamicVtableWrapper(object):

    cdef void *vtable_pointer
    cdef public object vtable

    def __init__(self, vtable):
        self.vtable = vtable
        self.write_vtable_pointer(vtable)

    def get_vtable_pointer(self):
        # Return a vtable **
        return <Py_uintptr_t> &self.vtable_pointer

    def write_vtable_pointer(self, new_table):
        self.vtable_pointer = <void *> <Py_uintptr_t> self.vtable.table_ptr

    def replace_vtable(self, new_vtable):
        old_vtable = self.vtable
        self.vtable = new_vtable
        self.write_vtable_pointer(new_vtable)

        # TODO: Manage keep alive of old_vtable until all consumers are done
        # TODO: with it

#------------------------------------------------------------------------
# Create Extension Type
#------------------------------------------------------------------------

def create_new_extension_type(metacls, name, bases, classdict,
                              exttype, vtable_wrapper):
    """
    Create an extension type from the given name, bases and classdict. Also
    takes a vtab struct minitype, and a attr_struct_type describing the
    object attributes.

    For static extension types:

            {
                PyObject_HEAD
                ...
                vtable *
                attribute1
                   ...
                attributeN
            }

        * The vtable is a contiguous block of memory holding function pointer

        * Attributes are put directly in the object

            * base classes must be compatible to allow a static ordering
              (we don't do C++-like dynamic offsets)

    For dynamic extension types:

            {
                PyObject_HEAD
                ...
                hash-based vtable *
                hash-based attribute table *
                attribute1
                   ...
                attributeN
            }

        * The vtable is a perfect hash-based vtable

        * Attributes are put directly in the object, BUT consumers may
          not rely on any ordering

              * Attribute locations are determined by a perfect hash-based
                table holding pointers to the attributes
    """
    cdef PyTypeObject *extclass_p
    cdef Py_ssize_t vtab_offset, attrs_offset

    orig_new = classdict.get('__new__', None)
    if orig_new:
        #
        orig_new = orig_new.__func__ # staticmethod is not callable!

    # __________________________________________________________________
    # Extension type constructor

    def new(cls, *args, **kwds):
        "Create a new object and patch it with a vtab"
        cdef PyObject *obj_p
        cdef void **vtab_location

        if orig_new is not None:
            new_func = orig_new
        else:
            assert issubclass(cls, extclass), (cls, extclass)
            new_func = super(extclass, cls).__new__

        if base_is_object:
            # Avoid warnings in py2.6 and errors in py3.x:
            #     DeprecationWarning: object.__new__() takes no parameters
            obj = new_func(cls)
        else:
            obj = new_func(cls, *args, **kwds)

        if (cls.__numba_vtab is not extclass.__numba_vtab or
                not isinstance(obj, cls)):
            # Subclass will set the vtab and attributes
            return obj

        # It is our responsibility to set the vtab and the ctypes attributes
        # Other fields are 0/NULL
        obj_p = <PyObject *> obj

        vtab_location = <void **> ((<char *> obj_p) + vtab_offset)
        vtab_location[0] = <void *> <Py_uintptr_t> cls.__numba_vtab_p

        attrs_pointer = (<Py_uintptr_t> obj_p) + attrs_offset
        obj._numba_attrs = ctypes.cast(attrs_pointer,
                                       cls.__numba_attributes_ctype)[0]

        return obj

    # __________________________________________________________________
    # Create extension type

    classdict['__new__'] = staticmethod(new)
    extclass = metacls(name, bases, classdict)
    assert isinstance(extclass, type)

    superclass_new = super(extclass, extclass).__new__
    cdef bint base_is_object = superclass_new.__self__ is object

    extclass_p = <PyTypeObject *> extclass

    # __________________________________________________________________
    # Update extension type

    attr_struct_type = exttype.attribute_table.to_struct()
    attr_struct_ctype = attr_struct_type.to_ctypes()

    vtab_offset = compute_vtab_offset(extclass)
    attrs_offset = compute_attrs_offset(extclass)

    upper = attrs_offset + ctypes.sizeof(attr_struct_ctype)
    upper = align(upper, 8)

    # print 'basicsize/vtab_offset/upper', extclass_p.tp_basicsize, lower, upper
    extclass_p.tp_basicsize = upper
    if extclass_p.tp_itemsize:
        raise NotImplementedError("Subclassing variable sized objects")

    # TODO: Figure out ctypes policy on tp_dealloc/tp_traverse/tp_clear
    # TODO: Don't use ctypes at all, but generate descriptors directly
    #if exttype.need_tp_dealloc:
    #    extclass_p.tp_dealloc = <destructor> tp_dealloc
    #    extclass_p.tp_clear = tp_clear
    #    extclass_p.tp_traverse = tp_traverse

    extclass.__numba_vtab_offset = vtab_offset
    extclass.__numba_obj_end = upper
    extclass.__numba_attr_offset = attrs_offset

    # Keep vtable alive
    if vtable_wrapper is None:
        # vtable_wrapper is None e.g. for the closure scope, which does
        # not need methods
        extclass.__numba_vtab = None
        extclass.__numba_vtab_p = 0
    else:
        extclass.__numba_vtab = vtable_wrapper
        extclass.__numba_vtab_p = vtable_wrapper.get_vtable_pointer()

    # __________________________________________________________________
    # Set ctypes attribute struct type, such that __new__ can create
    # _numba_attrs

    # TODO: Remove ctypes dependency and generate conversion descriptors

    extclass.__numba_attributes_ctype = attr_struct_type.pointer().to_ctypes()

    # __________________________________________________________________
    # Set exttype attribute

    extclass.__numba_ext_type = exttype
    extclass.exttype = exttype

    # __________________________________________________________________
    # Set offsets to objects (for tp_traverse, tp_clear, etc)

    offsets = getattr(extclass, '__numba_object_offset', [])
    offsets = offsets + compute_object_offsets(exttype, attrs_offset)
    extclass.__numba_object_offset = offsets

    return extclass

