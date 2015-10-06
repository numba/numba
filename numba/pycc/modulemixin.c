/*
 * This C file is compiled and linked into pycc-generated shared objects.
 * It provides the Numba helper functions for runtime use in pycc-compiled
 * functions.
 */

#include <Python.h>

#define NUMBA_EXPORT_FUNC(_rettype) _rettype
#define NUMBA_EXPORT_DATA(_vartype) _vartype

#include "../_helperlib.c"


/* XXX need to call _import_array() */
