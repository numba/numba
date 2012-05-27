cdef extern from "extensibletype.h":
    ctypedef struct PyExtensibleTypeObjectEntry:
        unsigned long id
        void *data

    type PyExtensibleType_Import()

    int PyCustomSlots_Check(obj)
    Py_ssize_t PyCustomSlots_Count(obj)
    PyExtensibleTypeObjectEntry PyCustomSlots_Table(obj)

    void *PyCustomSlots_Find(obj, unsigned long id, unsigned long mask)
