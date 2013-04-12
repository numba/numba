#include <Python.h>

static PyObject*
memoryview_get_buffer(PyObject *self, PyObject *args){
    PyObject *mv;
    if (!PyArg_ParseTuple(args, "O", &mv))
        return NULL;

    if (!PyMemoryView_Check(mv))
        return NULL;

    Py_buffer* buf = PyMemoryView_GET_BUFFER(mv);
    return PyLong_FromVoidPtr(buf->buf);
}

/** 
 * Gets a half-open range [start, end) which contains the array data
 * Modified from numpy https://github.com/numpy/numpy/blob/master/numpy/core/src/multiarray/array_assign.c
 */
static PyObject*
memoryview_get_extents(PyObject *self, PyObject *args)
{
    PyObject *mv;
    if (!PyArg_ParseTuple(args, "O", &mv))
        return NULL;

    if (!PyMemoryView_Check(mv))
        return NULL;

    Py_buffer* buf = PyMemoryView_GET_BUFFER(mv);

    Py_ssize_t start, end;
    int idim, ndim = buf->ndim;
    Py_ssize_t *dimensions = buf->shape,
               *strides = buf->strides;

    if (ndim < 0 ){
        PyErr_SetString(PyExc_ValueError, "buffer ndim < 0");
        return NULL;
    }

    if (!dimensions) {
        if (ndim == 0) {
            start = end = (Py_ssize_t) buf->buf;
            end += buf->itemsize;
            return Py_BuildValue("nn", start, end);
        }
        PyErr_SetString(PyExc_ValueError, "buffer shape is not defined");
        return NULL;
    }

    if (!strides) {
        PyErr_SetString(PyExc_ValueError, "buffer strides is not defined");
        return NULL;
    }

    PyObject *ret = NULL;
    /* Calculate with a closed range [start, end] */
    start = end = (Py_ssize_t)buf->buf;
    for (idim = 0; idim < ndim; ++idim) {
        Py_ssize_t stride = strides[idim], dim = dimensions[idim];
        /* If the array size is zero, return an empty range */
        if (dim == 0) {
            start = end = (Py_ssize_t)buf->buf;
            ret = Py_BuildValue("nn", start, end);
            break;
        }
        /* Expand either upwards or downwards depending on stride */
        else {
            if (stride > 0) {
                end += stride * (dim - 1);
            }
            else if (stride < 0) {
                start += stride * (dim - 1);
            }
        }
    }

    if (!ret) {
        /* Return a half-open range */
        Py_ssize_t out_start = start;
        Py_ssize_t out_end = end + buf->itemsize;

        ret = Py_BuildValue("nn", out_start, out_end);
    }

    return ret;
}

static PyMethodDef core_methods[] = {
#define declmethod(func) { #func , ( PyCFunction )func , METH_VARARGS , NULL }
    declmethod(memoryview_get_buffer),
    declmethod(memoryview_get_extents),
    { NULL },
#undef declmethod
};


// Module main function, hairy because of py3k port

#if (PY_MAJOR_VERSION >= 3)
    struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "mviewbuf",
        NULL,
        -1,
        core_methods,
        NULL, NULL, NULL, NULL
    };
#define INITERROR return NULL
    PyObject *
    PyInit_mviewbuf(void)
#else
#define INITERROR return
    PyMODINIT_FUNC
    initmviewbuf(void)
#endif
    {
#if PY_MAJOR_VERSION >= 3
        PyObject *module = PyModule_Create( &module_def );
#else
        PyObject *module = Py_InitModule("mviewbuf", core_methods);
#endif
        if (module == NULL)
            INITERROR;
#if PY_MAJOR_VERSION >= 3
        
        return module;
#endif
    }

