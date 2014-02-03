#include "_pymodule.h"
#include <stdio.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

static
int adapt_ndarray(PyObject *obj, void* arystruct) {
    PyArrayObject *ndary;
    int ndim;
    npy_intp *dims;
    npy_intp *strides;
    void *data;
    void **dataptr;
    npy_intp *dimsptr;
    npy_intp *stridesptr;
    int i;

    if (!PyArray_Check(obj)) {
        return -1;
    }

    ndary = (PyArrayObject*)obj;
    ndim = PyArray_NDIM(ndary);
    dims = PyArray_DIMS(ndary);
    strides = PyArray_STRIDES(ndary);
    data = PyArray_DATA(ndary);

    dataptr = (void**)arystruct;
    dimsptr = (npy_intp*)(dataptr + 1);
    stridesptr = dimsptr + ndim;

    for (i = 0; i < ndim; ++i) {
        dimsptr[i] = dims[i];
        stridesptr[i] = strides[i];
    }
    *dataptr = data;

    return 0;
}

static
PyObject* get_ndarray_adaptor(PyObject* self, PyObject *args)
{
    return PyLong_FromVoidPtr(&adapt_ndarray);
}

static PyMethodDef ext_methods[] = {
#define declmethod(func) { #func , ( PyCFunction )func , METH_VARARGS , NULL }
    declmethod(get_ndarray_adaptor),
    { NULL },
#undef declmethod
};


MOD_INIT(_numpyadapt)
{
    PyObject *m;

    import_array();

    MOD_DEF(m, "_numpyadapt", "No docs", ext_methods)

    if (m == NULL)
        return MOD_ERROR_VAL;

    return MOD_SUCCESS_VAL(m);
}
