/*
 * Definition of NRT functions for marshalling from / to Python objects.
 * This module is included by _nrt_pythonmod.c and by pycc-compiled modules.
 */

#include "../../_pymodule.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>
#include <numpy/arrayscalars.h>

#include "../../_arraystruct.h"
#include "../../_numba_common.h"
#include "nrt.h"


/*
 * Create a NRT MemInfo for data owned by a PyObject.
 */

static void
pyobject_dtor(void *ptr, size_t size, void* info) {
    PyGILState_STATE gstate;
    PyObject *ownerobj = info;

    gstate = PyGILState_Ensure();   /* ensure the GIL */
    Py_DECREF(ownerobj);            /* release the python object */
    PyGILState_Release(gstate);     /* release the GIL */
}

NUMBA_EXPORT_FUNC(NRT_MemInfo *)
NRT_meminfo_new_from_pyobject(void *data, PyObject *ownerobj) {
    size_t dummy_size = 0;
    Py_INCREF(ownerobj);
    return NRT_MemInfo_new(data, dummy_size, pyobject_dtor, ownerobj);
}


/*
 * A Python object wrapping a NRT meminfo.
 */

typedef struct {
    PyObject_HEAD
    NRT_MemInfo *meminfo;
} MemInfoObject;


static
int MemInfo_init(MemInfoObject *self, PyObject *args, PyObject *kwds) {
    static char *keywords[] = {"ptr", NULL};
    PyObject *raw_ptr_obj;
    void *raw_ptr;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", keywords, &raw_ptr_obj)) {
        return -1;
    }
    raw_ptr = PyLong_AsVoidPtr(raw_ptr_obj);
    if(PyErr_Occurred()) return -1;
    self->meminfo = (NRT_MemInfo *)raw_ptr;
    assert (NRT_MemInfo_refcount(self->meminfo) > 0 && "0 refcount");
    return 0;
}


static int
MemInfo_getbuffer(PyObject *exporter, Py_buffer *view, int flags) {
    Py_ssize_t len;
    void *buf;
    int readonly = 0;

    MemInfoObject *miobj = (MemInfoObject*)exporter;
    NRT_MemInfo *mi = miobj->meminfo;

    buf = NRT_MemInfo_data(mi);
    len = NRT_MemInfo_size(mi);
    return PyBuffer_FillInfo(view, exporter, buf, len, readonly, flags);
}

static PyBufferProcs MemInfo_bufferProcs = {MemInfo_getbuffer, NULL};

static
PyObject*
MemInfo_acquire(MemInfoObject *self) {
    NRT_MemInfo_acquire(self->meminfo);
    Py_RETURN_NONE;
}

static
PyObject*
MemInfo_release(MemInfoObject *self) {
    NRT_MemInfo_release(self->meminfo);
    Py_RETURN_NONE;
}

static
PyObject*
MemInfo_get_data(MemInfoObject *self, void *closure) {
    return PyLong_FromVoidPtr(NRT_MemInfo_data(self->meminfo));
}

static
PyObject*
MemInfo_get_refcount(MemInfoObject *self, void *closure) {
    size_t refct = NRT_MemInfo_refcount(self->meminfo);
    if ( refct == (size_t)-1 ) {
        PyErr_SetString(PyExc_ValueError, "invalid MemInfo");
        return NULL;
    }
    return PyLong_FromSize_t(refct);
}

static void
MemInfo_dealloc(MemInfoObject *self)
{
    NRT_MemInfo_release(self->meminfo);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyMethodDef MemInfo_methods[] = {
    {"acquire", (PyCFunction)MemInfo_acquire, METH_NOARGS,
     "Increment the reference count"
    },
    {"release", (PyCFunction)MemInfo_release, METH_NOARGS,
     "Decrement the reference count"
    },
    {NULL}  /* Sentinel */
};


static PyGetSetDef MemInfo_getsets[] = {
    {"data",
     (getter)MemInfo_get_data, NULL,
     "Get the data pointer as an integer",
     NULL},
    {"refcount",
     (getter)MemInfo_get_refcount, NULL,
     "Get the refcount",
     NULL},
    {NULL}  /* Sentinel */
};


static PyTypeObject MemInfoType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_nrt_python._MemInfo",                   /* tp_name*/
    sizeof(MemInfoObject),                    /* tp_basicsize*/
    0,                                        /* tp_itemsize*/
    (destructor)MemInfo_dealloc,              /* tp_dealloc*/
    0,                                        /* tp_print*/
    0,                                        /* tp_getattr*/
    0,                                        /* tp_setattr*/
    0,                                        /* tp_compare*/
    0,                                        /* tp_repr*/
    0,                                        /* tp_as_number*/
    0,                                        /* tp_as_sequence*/
    0,                                        /* tp_as_mapping*/
    0,                                        /* tp_hash */
    0,                                        /* tp_call*/
    0,                                        /* tp_str*/
    0,                                        /* tp_getattro*/
    0,                                        /* tp_setattro*/
    &MemInfo_bufferProcs,                      /* tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,  /* tp_flags*/
    0,                                        /* tp_doc */
    0,                                        /* tp_traverse */
    0,                                        /* tp_clear */
    0,                                        /* tp_richcompare */
    0,                                        /* tp_weaklistoffset */
    0,                                        /* tp_iter */
    0,                                        /* tp_iternext */
    MemInfo_methods,                          /* tp_methods */
    0,                                        /* tp_members */
    MemInfo_getsets,                          /* tp_getset */
    0,                                        /* tp_base */
    0,                                        /* tp_dict */
    0,                                        /* tp_descr_get */
    0,                                        /* tp_descr_set */
    0,                                        /* tp_dictoffset */
    (initproc)MemInfo_init,                   /* tp_init */
    0,                                        /* tp_alloc */
    0,                                        /* tp_new */
};


/*
Return a MemInfo* as a MemInfoObject*
The NRT reference to the MemInfo is borrowed.
*/
NUMBA_EXPORT_FUNC(MemInfoObject*)
NRT_meminfo_as_pyobject(NRT_MemInfo *meminfo) {
    MemInfoObject *mi;
    PyObject *addr;

    addr = PyLong_FromVoidPtr(meminfo);
    if (!addr) return NULL;
    mi = (MemInfoObject*)PyObject_CallFunctionObjArgs((PyObject *)&MemInfoType, addr, NULL);
    Py_DECREF(addr);
    if (!mi) return NULL;
    return mi;
}


/*
Return a MemInfo* from a MemInfoObject*
A new reference is returned.
*/
NUMBA_EXPORT_FUNC(NRT_MemInfo*)
NRT_meminfo_from_pyobject(MemInfoObject *miobj) {
    NRT_MemInfo_acquire(miobj->meminfo);
    return miobj->meminfo;
}


/*
 * Array adaptor code
 */

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
 * This function was renamed in 0.52.0 to specify that it acquires references.
 * It used to steal the reference of the arystruct.
 * Refer to https://github.com/numba/numba/pull/6446
 */
NUMBA_EXPORT_FUNC(PyObject *)
NRT_adapt_ndarray_to_python_acqref(arystruct_t* arystruct, int ndim,
                            int writeable, PyArray_Descr *descr)
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
        /*  Note: MemInfo_init() does not incref.  This function steals the
         *        NRT reference, which we need to acquire.
         */
        NRT_MemInfo_acquire(arystruct->meminfo);
        if (MemInfo_init(miobj, args, NULL)) {
            return NULL;
        }
        Py_DECREF(args);
    }

    shape = arystruct->shape_and_strides;
    strides = shape + ndim;
    Py_INCREF((PyObject *) descr);
    array = (PyArrayObject *) PyArray_NewFromDescr(&PyArray_Type, descr, ndim,
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


/* Initialization subroutines for modules including this source file */

static int
init_nrt_python_module(PyObject *module)
{
    MemInfoType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&MemInfoType))
        return -1;
    return 0;
}
