cdef double nan

from numpy import nan
cimport cython
from customslots cimport *

cdef extern from "thestandard.h":
    unsigned long EXTENSIBLETYPE_DOUBLE_FUNC_SLOT

# Import call_func from a seperately compiled C file to avoid inlining it

ctypedef double (*funcptr_t)(double)
cdef extern:
    double call_func(object)

def sum_lookups(obj, int n):
    # Do the lookup n times
    cdef int i
    cdef double s = 0
    for i in range(n):
        s += call_func(obj)
    return s

def sum_baseline(obj, int n):
    # Do the same work as sum_lookups, but only do the lookup once
    cdef double s = 0
    cdef PyCustomSlot *slot = PyCustomSlots_Find(
            obj, EXTENSIBLETYPE_DOUBLE_FUNC_SLOT, 0)
    if slot == NULL:
        return 0.0
    cdef funcptr_t funcptr = <funcptr_t>slot.data.pointer
    for i in range(n):
        s += funcptr(3.0)
    return s
