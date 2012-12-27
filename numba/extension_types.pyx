"""
This modules creates extension types.

See also numba.extension_type_inference
"""

cimport cython
from cpython cimport PyObject

import sys
import ctypes

import numba

ctypedef object (*tp_new_func)(PyObject *, PyObject *, PyObject *)
ctypedef void (*destructor)(PyObject *)
ctypedef int (*visitproc)(PyObject *, void *)

cdef extern size_t closure_field_offset
cdef extern int NumbaFunction_init() except -1
cdef extern object NumbaFunction_NewEx(
                PyMethodDef *ml, module, code, PyObject *closure,
                void *native_func, native_signature, keep_alive)

cdef extern from *:
    ctypedef unsigned long Py_uintptr_t

    ctypedef struct PyTypeObject:
        tp_new_func tp_new
        destructor tp_dealloc
        int (*tp_traverse)(PyObject *o, visitproc v, void *a)
        int (*tp_clear)(PyObject *o)

        long tp_dictoffset
        Py_ssize_t tp_itemsize
        Py_ssize_t tp_basicsize
        PyTypeObject *tp_base

    ctypedef struct PyMethodDef:
        pass

    void Py_XDECREF(PyObject *)
    void Py_INCREF(PyObject *)
    void Py_CLEAR(PyObject *)

NumbaFunction_init()
NumbaFunction_NewEx_pointer = <Py_uintptr_t> &NumbaFunction_NewEx

numbafunc_closure_field_offset = closure_field_offset

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
    for field_name, field_type in ext_type.attribute_struct.fields:
        if field_type.is_object or field_type.is_array:
            offset = ext_type.attribute_struct.offsetof(field_name)
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
# Create Extension Type
#------------------------------------------------------------------------

def create_new_extension_type(name, bases, dict, ext_numba_type,
                              vtab, vtab_type, llvm_methods, method_pointers):
    """
    Create an extension type from the given name, bases and dict. Also
    takes a vtab struct minitype, and a struct_type describing the
    object attributes.
    """
    cdef PyTypeObject *ext_type_p
    cdef Py_ssize_t vtab_offset, attrs_offset

    orig_new = dict.get('__new__', None)
    def new(cls, *args, **kwds):
        "Create a new object and patch it with a vtab"
        cdef PyObject *obj_p
        cdef void **vtab_location

        if orig_new is not None:
            obj = orig_new(cls, *args, **kwds)
        else:
            assert issubclass(cls, ext_type), (cls, ext_type)
            obj = super(ext_type, cls).__new__(cls, *args, **kwds)

        if (cls.__numba_vtab is not ext_type.__numba_vtab or
                not isinstance(obj, cls)):
            # Subclass will set the vtab and attributes
            return obj

        # It is our responsibility to set the vtab and the ctypes attributes
        # Other fields are 0/NULL
        obj_p = <PyObject *> obj

        vtab_location = <void **> ((<char *> obj_p) + vtab_offset)
        if vtab:
            vtab_location[0] = <void *> <Py_uintptr_t> cls.__numba_vtab_p
        else:
            vtab_location[0] = NULL

        attrs_pointer = (<Py_uintptr_t> obj_p) + attrs_offset
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
    struct_type = ext_numba_type.attribute_struct
    struct_ctype = struct_type.to_ctypes()
    vtab_offset = compute_vtab_offset(ext_type)
    attrs_offset = compute_attrs_offset(ext_type)
    upper = attrs_offset + ctypes.sizeof(struct_ctype)
    upper = align(upper, 8)

    # print 'basicsize/vtab_offset/upper', ext_type_p.tp_basicsize, lower, upper
    ext_type_p.tp_basicsize = upper
    if ext_type_p.tp_itemsize:
        raise NotImplementedError("Subclassing variable sized objects")

    # TODO: Figure out ctypes policy on tp_dealloc/tp_traverse/tp_clear
    # TODO: Don't use ctypes at all, but generate descriptors directly
    #if ext_numba_type.need_tp_dealloc:
    #    ext_type_p.tp_dealloc = <destructor> tp_dealloc
    #    ext_type_p.tp_clear = tp_clear
    #    ext_type_p.tp_traverse = tp_traverse

    ext_type.__numba_vtab_offset = vtab_offset
    ext_type.__numba_obj_end = upper
    ext_type.__numba_attr_offset = attrs_offset

    ext_type.__numba_vtab = vtab
    ext_type.__numba_vtab_type = vtab_type

    if vtab:
        vtab_p = ctypes.byref(vtab)
        ext_type.__numba_vtab_p = ctypes.cast(vtab_p, ctypes.c_void_p).value
    else:
        ext_type.__numba_vtab_p = None

    ext_type.__numba_orig_tp_new = <Py_uintptr_t> ext_type_p.tp_new
    ext_type.__numba_struct_type = struct_type
    ext_type.__numba_struct_ctype_p = struct_type.pointer().to_ctypes()
    ext_type.__numba_lfuncs = llvm_methods
    ext_type.__numba_method_pointers = method_pointers
    ext_type.__numba_ext_type = ext_numba_type
    ext_type.exttype = ext_numba_type

    offsets = getattr(ext_type, '__numba_object_offset', [])
    offsets = offsets + compute_object_offsets(ext_numba_type, attrs_offset)
    ext_type.__numba_object_offset = offsets

    return ext_type

#------------------------------------------------------------------------
# Create Numba Functions (numbafunction.c)
#------------------------------------------------------------------------

def create_function(methoddef, py_func, lfunc_pointer, signature, modname):
    cdef Py_uintptr_t methoddef_p = ctypes.cast(ctypes.byref(methoddef),
                                                ctypes.c_void_p).value
    cdef PyMethodDef *ml = <PyMethodDef *> methoddef_p
    cdef Py_uintptr_t lfunc_p = lfunc_pointer

    result = NumbaFunction_NewEx(ml, modname, getattr(py_func, "func_code", None),
                                 NULL, <void *> lfunc_p, signature, py_func)
    return result
