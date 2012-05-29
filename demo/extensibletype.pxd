cdef extern from "extensibletype.h":
    ctypedef struct PyCustomSlot:
        unsigned long id
        void *data

    type PyExtensibleType_Import()

    int PyCustomSlots_Check(obj)
    Py_ssize_t PyCustomSlots_Count(obj)
    PyCustomSlot *PyCustomSlots_Table(obj)

    ctypedef unsigned long uint32_t

    PyCustomSlot *PyCustomSlots_Find(obj, uint32_t id, Py_ssize_t start)
