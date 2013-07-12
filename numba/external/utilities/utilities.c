/*
    C utility functions
*/

#include <Python.h>
#include "_numba.h"

#define EXPORT_FUNCTION(func, module, errlabel) {                            \
    PyObject *func_val = PyLong_FromUnsignedLongLong((Py_uintptr_t) &func);  \
    if (!func_val)                                                           \
        goto errlabel;                                                       \
    if (PyModule_AddObject(module, #func, func_val) < 0)                     \
        goto errlabel;                                                       \
    }

#include "type_conversion.c"
#include "virtuallookup.c"
#include "cpyutils.c"

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    __Numba_NAMESTR("utilities"),
    0,      /* m_doc */
    -1,     /* m_size */
    NULL,   /* m_methods */
    NULL,   /* m_reload */
    NULL,   /* m_traverse */
    NULL,   /* m_clear */
    NULL    /* m_free */
};
#endif

#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC initutilities(void)
#else
PyMODINIT_FUNC PyInit_utilities(void)
#endif
{
    PyObject *module;
    PyDateTime_IMPORT;

#if PY_MAJOR_VERSION < 3
    module = Py_InitModule4(__Numba_NAMESTR("utilities"), NULL, 0, 0, PYTHON_API_VERSION);
    Py_XINCREF(module);
#else
    module = PyModule_Create(&moduledef);
#endif

    if (!module)
        goto error;

    /* Call all export functions */
    if (export_type_conversion(module) < 0)
        goto error;
    if (export_virtuallookup(module) < 0)
        goto error;
    if (export_cpyutils(module) < 0)
        goto error;

    goto success; /* done */

error:
    Py_XDECREF(module);
    module = NULL;
success:

#if PY_MAJOR_VERSION < 3
    return;
#else
    return module;
#endif
}

