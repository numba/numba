#ifndef NUMBA_EXTENSION_HELPER_H_
#define NUMBA_EXTENSION_HELPER_H_

#include "Python.h"
#include "../_numba_common.h"

/* Define all runtime-required symbols in this C module, but do not
   export them outside the shared library if possible. */
#define NUMBA_EXPORT_FUNC(_rettype) VISIBILITY_HIDDEN _rettype
#define NUMBA_EXPORT_DATA(_vartype) VISIBILITY_HIDDEN _vartype

NUMBA_EXPORT_FUNC(Py_ssize_t)
aligned_size(Py_ssize_t sz);

#include "dictobject.h"
#include "listobject.h"

#endif // end NUMBA_EXTENSION_HELPER_H_
