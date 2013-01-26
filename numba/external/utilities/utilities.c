/*
    C utility functions
*/

#include <Python.h>

#define EXPORT_FUNCTION(func, module, errlabel) {                            \
    PyObject *func_val = PyLong_FromUnsignedLongLong((Py_uintptr_t) &func);  \
    if (!func_val)                                                           \
        goto errlabel;                                                       \
    if (PyModule_AddObject(module, #func, func_val) < 0)                     \
        goto errlabel;                                                       \
    }
