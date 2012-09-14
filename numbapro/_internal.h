typedef struct {
    PyUFuncObject ufunc;
    PyUFuncObject *ufunc_original;
    PyObject *minivect_dispatcher;
    PyObject *cuda_dispatcher;
} PyDynUFuncObject;