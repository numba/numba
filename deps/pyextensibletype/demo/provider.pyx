from cpython cimport PyObject

cdef extern from "provider_c_code.h":
    int ProviderType_Ready()
    PyObject Provider_Type

ProviderType_Ready()
Provider = <object><PyObject*>&Provider_Type
