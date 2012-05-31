cdef extern from "customslots.h":
    ctypedef unsigned long uintptr_t

    ctypedef union pyx_data:
        void *pointer
        Py_ssize_t objoffset
        uintptr_t flags
    
    ctypedef struct PyCustomSlot:
        uintrptr_t id
        pyx_data data

    int PyCustomSlots_Check(obj)
    Py_ssize_t PyCustomSlots_Count(obj)
    PyCustomSlot *PyCustomSlots_Table(obj)

    PyCustomSlot *PyCustomSlots_Find(obj, uintptr_t id, Py_ssize_t start)
