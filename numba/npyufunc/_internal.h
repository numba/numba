/* Python include */

#ifndef NUMBA_UFUNC_INTERNAL_H_
#define NUMBA_UFUNC_INTERNAL_H_

#include "../_pymodule.h"
#include <structmember.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/ndarrayobject.h"
#include "numpy/ufuncobject.h"

extern PyObject *ufunc_fromfunc(PyObject *NPY_UNUSED(dummy), PyObject *args);

int PyUFunc_GeneralizedFunction(PyUFuncObject *ufunc,
                                PyObject *args, PyObject *kwds,
                                PyArrayObject **op);

#define APPEND_(X, Y) X #Y
#define APPEND(X, Y) APPEND_(X, Y)
#define SENTRY_VALID_LONG(X) if( (X) == -1 ){                        \
    PyErr_SetString(PyExc_RuntimeError,                              \
                    APPEND("PyLong_AsLong overflow at ", __LINE__)); \
    return NULL;                                                     \
}

#endif  /* NUMBA_UFUNC_INTERNAL_H_ */