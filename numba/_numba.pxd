from cpython cimport PyObject
from cpython.tuple cimport (PyTuple_New,
                            PyTuple_GET_ITEM,
                            PyTuple_SET_ITEM,
                            PyTuple_GET_SIZE)

cdef extern from *:
    ctypedef unsigned long Py_uintptr_t

    void Py_XDECREF(PyObject *)
    void Py_INCREF(PyObject *)
    void Py_CLEAR(PyObject *)

    object PyObject_Call(PyObject *callable_object,
                         PyObject *args, PyObject *kw)