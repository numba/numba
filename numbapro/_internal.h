/* Python include */
#include "Python.h"
#include <structmember.h>

/* #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION */
#include "numpy/ndarrayobject.h"
#include "numpy/ufuncobject.h"

typedef struct {
    PyUFuncObject ufunc;
    PyUFuncObject *ufunc_original;
    PyObject *minivect_dispatcher;
    PyObject *cuda_dispatcher;
    int use_cuda_gufunc;
} PyDynUFuncObject;

extern PyTypeObject PyDynUFunc_Type;

extern PyObject *ufunc_fromfunc(PyObject *NPY_UNUSED(dummy), PyObject *args);
extern PyObject * ufunc_fromfuncsig(PyObject *NPY_UNUSED(dummy),
                                    PyObject *args);

PyObject *
PyDynUFunc_New(PyUFuncObject *ufunc, PyObject *minivect_dispatcher,
               PyObject *cuda_dispatcher, int use_cuda_gufunc);
int PyUFunc_GeneralizedFunction(PyUFuncObject *ufunc,
                                PyObject *args, PyObject *kwds,
                                PyArrayObject **op);


#if PY_MAJOR_VERSION >= 3
#define IS_PY3K
#endif

#define APPEND_(X, Y) X #Y
#define APPEND(X, Y) APPEND_(X, Y)
#define SENTRY_VALID_LONG(X) if( (X) == -1 ){                        \
    PyErr_SetString(PyExc_RuntimeError,                              \
                    APPEND("PyLong_AsLong overflow at ", __LINE__)); \
    return NULL;                                                     \
}

#ifdef IS_PY3K
#define INIT(name) PyObject *name(void)
#else
#define INIT(name) PyMODINIT_FUNC name(void)
#endif

INIT(init_ufunc);
INIT(init_gufunc);

typedef struct {
    PyObject_HEAD
    /* Pointer to the raw data buffer */
    char *data;
    /* The number of dimensions, also called 'ndim' */
    int nd;
    /* The size in each dimension, also called 'shape' */
    npy_intp *dimensions;
    /*
     * Number of bytes to jump to get to the
     * next element in each dimension
     */
    npy_intp *strides;
    /*
     * This object is decref'd upon
     * deletion of array. Except in the
     * case of UPDATEIFCOPY which has
     * special handling.
     *
     * For views it points to the original
     * array, collapsed so no chains of
     * views occur.
     *
     * For creation from buffer object it
     * points to an object that shold be
     * decref'd on deletion
     *
     * For UPDATEIFCOPY flag this is an
     * array to-be-updated upon deletion
     * of this one
     */
    PyObject *base;
    /* Pointer to type structure */
    PyArray_Descr *descr;
    /* Flags describing array -- see below */
    int flags;
    /* For weak references */
    PyObject *weakreflist;
    void *maskna_dtype;
    void *maskna_data;
    void *maskna_strides;
} ndarray;
