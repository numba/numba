cdef class somemeta(type):
    pass
    
#class A:cdef class foo(type):
#    pass

    

cdef extern from "extensibletype.h":
    type PyExtensibleType_GetMetaClass()

#cdef type PyExtensibleType = PyExtensibleType_GetMetaClass()

#cdef class bar:
#    __metaclass__ = PyExtensibleType

def f():
    return PyExtensibleType_GetMetaClass()
