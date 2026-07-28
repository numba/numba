#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>
#if NPY_ABI_VERSION >= 0x02000000
    #include <numpy/npy_2_compat.h>
#endif

#include "_nparraystruct.h"
#define NUMBA_EXPORT_FUNC(_rettype) static _rettype
#include "../core/runtime/_nrt_python.c"

/*
 * Array adaptor code
 */


NUMBA_EXPORT_FUNC(int)
numba_adapt_ndarray(PyObject *obj, arystruct_t* arystruct) {
    PyArrayObject *ndary;
    int i, ndim;
    npy_intp *p;

    if (!PyArray_Check(obj)) {
        return -1;
    }

    ndary = (PyArrayObject*)obj;
    ndim = PyArray_NDIM(ndary);

    arystruct->data = PyArray_DATA(ndary);
    arystruct->nitems = PyArray_SIZE(ndary);
    arystruct->itemsize = PyArray_ITEMSIZE(ndary);
    arystruct->parent = obj;
    p = arystruct->shape_and_strides;
    for (i = 0; i < ndim; i++, p++) {
        *p = PyArray_DIM(ndary, i);
    }
    for (i = 0; i < ndim; i++, p++) {
        *p = PyArray_STRIDE(ndary, i);
    }
    arystruct->meminfo = NULL;
    return 0;
}


static
PyObject* try_to_return_parent(arystruct_t *arystruct, int ndim,
                               PyArray_Descr *descr)
{
    int i;
    PyArrayObject *array = (PyArrayObject *)arystruct->parent;

    if (!PyArray_Check(arystruct->parent))
        /* Parent is a generic buffer-providing object */
        goto RETURN_ARRAY_COPY;

    if (PyArray_DATA(array) != arystruct->data)
        goto RETURN_ARRAY_COPY;

    if (PyArray_NDIM(array) != ndim)
        goto RETURN_ARRAY_COPY;

    if (PyObject_RichCompareBool((PyObject *) PyArray_DESCR(array),
                                 (PyObject *) descr, Py_EQ) <= 0)
        goto RETURN_ARRAY_COPY;

    for(i = 0; i < ndim; ++i) {
        if (PyArray_DIMS(array)[i] != arystruct->shape_and_strides[i])
            goto RETURN_ARRAY_COPY;
        if (PyArray_STRIDES(array)[i] != arystruct->shape_and_strides[ndim + i])
            goto RETURN_ARRAY_COPY;
    }

    /* Yes, it is the same array
       Return new reference */
    Py_INCREF((PyObject *)array);
    return (PyObject *)array;

RETURN_ARRAY_COPY:
    return NULL;
}

/**
 * This function is used during the boxing of ndarray type.
 * `arystruct` is a structure containing essential information from the
 *             unboxed array.
 * `retty` is the subtype of the NumPy PyArray_Type this function should return.
 *         This is related to `numba.core.types.Array.box_type`.
 * `ndim` is the number of dimension of the array.
 * `writeable` corresponds to the "writable" flag in NumPy ndarray.
 * `descr` is the NumPy data type description.
 *
 * This function was renamed in 0.52.0 to specify that it acquires references.
 * It used to steal the reference of the arystruct.
 * Refer to https://github.com/numba/numba/pull/6446
 */
NUMBA_EXPORT_FUNC(PyObject *)
NRT_adapt_ndarray_to_python_acqref(arystruct_t* arystruct, PyTypeObject *retty,
                            int ndim, int writeable, PyArray_Descr *descr)
{
    PyArrayObject *array;
    MemInfoObject *miobj = NULL;
    PyObject *args;
    npy_intp *shape, *strides;
    int flags = 0;

    if (descr == NULL) {
        PyErr_Format(PyExc_RuntimeError,
                     "In 'NRT_adapt_ndarray_to_python', 'descr' is NULL");
        return NULL;
    }

    if (!NUMBA_PyArray_DescrCheck(descr)) {
        PyErr_Format(PyExc_TypeError,
                     "expected dtype object, got '%.200s'",
                     Py_TYPE(descr)->tp_name);
        return NULL;
    }

    if (arystruct->parent) {
        PyObject *obj = try_to_return_parent(arystruct, ndim, descr);
        if (obj) {
            return obj;
        }
    }

    if (arystruct->meminfo) {
        /* wrap into MemInfoObject */
        miobj = PyObject_New(MemInfoObject, &MemInfoType);
        args = PyTuple_New(1);
        /* SETITEM steals reference */
        PyTuple_SET_ITEM(args, 0, PyLong_FromVoidPtr(arystruct->meminfo));
        NRT_Debug(nrt_debug_print("NRT_adapt_ndarray_to_python arystruct->meminfo=%p\n", arystruct->meminfo));
        /*  Note: MemInfo_init() does not incref.  This function steals the
         *        NRT reference, which we need to acquire.
         */
        NRT_Debug(nrt_debug_print("NRT_adapt_ndarray_to_python_acqref created MemInfo=%p\n", miobj));
        NRT_MemInfo_acquire(arystruct->meminfo);
        if (MemInfo_init(miobj, args, NULL)) {
            NRT_Debug(nrt_debug_print("MemInfo_init failed.\n"));
            return NULL;
        }
        Py_DECREF(args);
    }

    shape = arystruct->shape_and_strides;
    strides = shape + ndim;
    Py_INCREF((PyObject *) descr);
    array = (PyArrayObject *) PyArray_NewFromDescr(retty, descr, ndim,
                                                   shape, strides, arystruct->data,
                                                   flags, (PyObject *) miobj);

    if (array == NULL)
        return NULL;

    /* Set writable */
#if NPY_API_VERSION >= 0x00000007
    if (writeable) {
        PyArray_ENABLEFLAGS(array, NPY_ARRAY_WRITEABLE);
    }
    else {
        PyArray_CLEARFLAGS(array, NPY_ARRAY_WRITEABLE);
    }
#else
    if (writeable) {
        array->flags |= NPY_WRITEABLE;
    }
    else {
        array->flags &= ~NPY_WRITEABLE;
    }
#endif

    if (miobj) {
        /* Set the MemInfoObject as the base object */
#if NPY_API_VERSION >= 0x00000007
        if (-1 == PyArray_SetBaseObject(array,
                                        (PyObject *) miobj))
        {
            Py_DECREF(array);
            Py_DECREF(miobj);
            return NULL;
        }
#else
        PyArray_BASE(array) = (PyObject *) miobj;
#endif

    }
    return (PyObject *) array;
}

NUMBA_EXPORT_FUNC(void)
NRT_adapt_buffer_from_python(Py_buffer *buf, arystruct_t *arystruct)
{
    int i;
    npy_intp *p;

    if (buf->obj) {
        /* Allocate new MemInfo only if the buffer has a parent */
        arystruct->meminfo = NRT_meminfo_new_from_pyobject((void*)buf->buf, buf->obj);
    }
    arystruct->data = buf->buf;
    arystruct->itemsize = buf->itemsize;
    arystruct->parent = buf->obj;
    arystruct->nitems = 1;
    p = arystruct->shape_and_strides;
    for (i = 0; i < buf->ndim; i++, p++) {
        *p = buf->shape[i];
        arystruct->nitems *= buf->shape[i];
    }
    for (i = 0; i < buf->ndim; i++, p++) {
        *p = buf->strides[i];
    }
}

NUMBA_EXPORT_FUNC(int)
NRT_adapt_ndarray_from_python(PyObject *obj, arystruct_t* arystruct) {
    PyArrayObject *ndary;
    int i, ndim;
    npy_intp *p;
    void *data;

    if (!PyArray_Check(obj)) {
        return -1;
    }

    ndary = (PyArrayObject*)obj;
    ndim = PyArray_NDIM(ndary);
    data = PyArray_DATA(ndary);

    arystruct->meminfo = NRT_meminfo_new_from_pyobject((void*)data, obj);
    arystruct->data = data;
    arystruct->nitems = PyArray_SIZE(ndary);
    arystruct->itemsize = PyArray_ITEMSIZE(ndary);
    arystruct->parent = obj;
    p = arystruct->shape_and_strides;
    for (i = 0; i < ndim; i++, p++) {
        *p = PyArray_DIM(ndary, i);
    }
    for (i = 0; i < ndim; i++, p++) {
        *p = PyArray_STRIDE(ndary, i);
    }

    NRT_Debug(nrt_debug_print("NRT_adapt_ndarray_from_python %p\n",
                              arystruct->meminfo));
    return 0;
}

static
void* wrap_import_array(void) {
    import_array(); /* import array returns NULL on failure */
    return (void*)1;
}


static
int init_numpy(void) {
    return wrap_import_array() != NULL;
}



