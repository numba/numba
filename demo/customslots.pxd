cdef extern from "customslots.h":
    ctypedef struct PyCustomSlot:
        unsigned long id
        void *data

    int PyCustomSlots_Check(obj)
    Py_ssize_t PyCustomSlots_Count(obj)
    PyCustomSlot *PyCustomSlots_Table(obj)

    ctypedef unsigned long uintptr_t

    PyCustomSlot *PyCustomSlots_Find(obj, uintptr_t id, Py_ssize_t start)
