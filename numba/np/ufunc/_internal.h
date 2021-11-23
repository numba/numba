/* Python include */

#ifndef NUMBA_UFUNC_INTERNAL_H_
#define NUMBA_UFUNC_INTERNAL_H_

#include "../../_pymodule.h"
#include <structmember.h>
#include "../../cext/cext.h"
#include "Python.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/ndarrayobject.h"
#include "numpy/ufuncobject.h"
#include "../../_arraystruct.h"
#include "../../../numba/core/runtime/nrt.h"

extern PyObject *ufunc_fromfunc(PyObject *NPY_UNUSED(dummy), PyObject *args);

typedef struct {
  PyObject_HEAD
  PyObject      * dispatcher;
  PyUFuncObject * ufunc;
  PyObject      * keepalive;
  int             frozen;
} PyDUFuncObject;

NUMBA_EXPORT_FUNC(static PyObject *)
dufunc_reduce_direct(PyDUFuncObject * self, arystruct_t * args, int axis, PyTypeObject *retty,  PyArray_Descr *descr);

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
