/* Python include */
#include "Python.h"
#include <structmember.h>

#if PY_MAJOR_VERSION >= 3
  #define MOD_ERROR_VAL NULL
  #define MOD_SUCCESS_VAL(val) val
  #define MOD_INIT(name) PyMODINIT_FUNC PyInit_##name(void)
  #define MOD_DEF(ob, name, doc, methods) { \
          static struct PyModuleDef moduledef = { \
            PyModuleDef_HEAD_INIT, name, doc, -1, methods, }; \
          ob = PyModule_Create(&moduledef); }
  #define MOD_INIT_EXEC(name) PyInit_##name();
#else
  #define MOD_ERROR_VAL
  #define MOD_SUCCESS_VAL(val)
  #define MOD_INIT(name) PyMODINIT_FUNC init##name(void)
  #define MOD_DEF(ob, name, doc, methods) \
          ob = Py_InitModule3(name, methods, doc);
  #define MOD_INIT_EXEC(name) init##name();
#endif

/* #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION */
#include "numpy/ndarrayobject.h"
#include "numpy/ufuncobject.h"

typedef struct {
    PyUFuncObject ufunc;
    PyUFuncObject *ufunc_original;
    PyObject *dispatcher;
} PyDynUFuncObject;

extern PyTypeObject PyDynUFunc_Type;

extern PyObject *ufunc_fromfunc(PyObject *NPY_UNUSED(dummy), PyObject *args);

extern PyObject * ufunc_fromfuncsig(PyObject *NPY_UNUSED(dummy),
                                    PyObject *args);

PyObject *
PyDynUFunc_New(PyUFuncObject *ufunc, PyObject *dispatcher);

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


MOD_INIT(ufunc);
MOD_INIT(gufunc);
