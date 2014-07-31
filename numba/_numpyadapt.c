#include "_pymodule.h"
#include <stdio.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

#ifndef NPY_ARRAY_BEHAVED
    #define NPY_ARRAY_BEHAVED NPY_BEHAVED
#endif

static
int adapt_ndarray(PyObject *obj, void* arystruct) {
    PyArrayObject *ndary;
    int ndim;
    npy_intp *dims;
    npy_intp *strides;
    void *data;
    void **dataptr;
    void **objectptr;
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
    objectptr = stridesptr + ndim;

    for (i = 0; i < ndim; ++i) {
        dimsptr[i] = dims[i];
        stridesptr[i] = strides[i];
    }
    *dataptr = data;
    *objectptr = obj;

    return 0;
}

static
PyObject* ndarray_new(int nd,
                      npy_intp *dims,   /* shape */
                      npy_intp *strides,
                      void* data,
                      int type_num,
                      int itemsize)
{
    PyObject *ndary;
    int flags = NPY_ARRAY_BEHAVED;
    ndary = PyArray_New((PyTypeObject*)&PyArray_Type, nd, dims, type_num,
                       strides, data, 0, flags, NULL);
    return ndary;
}

static
PyObject* get_ndarray_adaptor(PyObject* self, PyObject *args)
{
    return PyLong_FromVoidPtr(&adapt_ndarray);
}

static
PyObject* get_ndarray_new(PyObject* self, PyObject *args)
{
    return PyLong_FromVoidPtr(&ndarray_new);
}


static PyMethodDef ext_methods[] = {
#define declmethod(func) { #func , ( PyCFunction )func , METH_VARARGS , NULL }
    declmethod(get_ndarray_adaptor),
    declmethod(get_ndarray_new),
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
