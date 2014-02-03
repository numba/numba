/* Utilities for use with the CPython C-API */

#include "exceptions.c"

static int
export_cpyutils(PyObject *module)
{
    EXPORT_FUNCTION(Raise, module, error)

    return 0;
error:
    return -1;
}