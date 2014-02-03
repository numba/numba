"""
This demo depicts how we can support overriding of numba methods in python.
In Cython, cpdef methods check to see if the object has a dict, and if so
look up the method at runtime and dispatch if found:

    cpdef mymethod(self):
        if hasattr(self, '__dict__'):
            return self.mymethod()

        <method body>

This is needed for calls that go through the vtable, and hence hit the
inherited Cython version, but not the Python one (see the 'obj.mymethod' call
of a typed object below in 'func').

We can use a metaclass to detect overridden methods and patch the vtable (
create a new one for each class). We can further detect monkey-patching on
the class and instances through descriptors:

    * monkey patches on the class mutate the vtable
    * monkey patches on the object create a new object-specific vtable
        - or perhaps are simply disallowed
"""

import ctypes

from cpython cimport PyObject

cdef extern from *:
    ctypedef unsigned int Py_uintptr_t

    struct __pyx_obj_4test_Base:
        void *__pyx_vtab

    ctypedef struct PyTypeObject:
        PyObject *ob_type

cdef mymethod_wrapper(self, int arg):
    self.mymethod(<object> arg)

class VTable(ctypes.Structure):
    _fields_ = [('mymethod', ctypes.c_void_p)]

cdef class MetaClass(type):

    cdef public object vtable

    def __init__(self, name, bases, dict):
        if "mymethod" in dict:
            mymethod_p = <Py_uintptr_t> &mymethod_wrapper
            self.vtable = VTable(ctypes.c_void_p(mymethod_p))
        else:
            self.vtable = None

# --- Test classes ---

cdef class Base(object):
    cdef mymethod(self, int arg):
        print "mymethod in Base", arg

(<PyTypeObject *> Base).ob_type = <PyObject *> MetaClass

class Derived(Base):
    def __init__(self):
        cdef Py_uintptr_t vtable_p

        vtable = type(self).vtable

        if vtable is not None:
            vtable_p = ctypes.addressof(vtable)
            (<__pyx_obj_4test_Base *> self).__pyx_vtab = <void *> vtable_p


    def mymethod(self, int arg):
        print "mymethod in Derived", arg


cdef func(Base obj):
    obj.mymethod(10)


func(Base()) # prints: mymethod in Base 10
func(Derived()) # prints: mymethod in Derived 10