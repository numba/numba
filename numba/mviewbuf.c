#include "_pymodule.h"

/**
 * A BufferProxy wraps a read-only buffer in a writable buffer. This is required
 * for NumPy void and record types, which only expose a read-only buffer. Note
 * that wrapping a read-only buffer may have some surprising effects on buffers
 * which expect the data from their read-only buffer not to be modified.
 */
typedef struct {
    PyObject_HEAD
    PyObject *wrapped;
    int ob_exports;
    Py_buffer wrapped_buf;
#if (PY_MAJOR_VERSION < 3)
    void *wrapped_ptr;
    Py_ssize_t wrapped_len;
    int ob_old_exports;
#endif
} BufferProxyObject;

static int
BufferProxyObject_getbuffer(BufferProxyObject *self, Py_buffer *view,
                            int flags) {
    int supported_flags = PyBUF_WRITABLE|PyBUF_ND|PyBUF_STRIDES|PyBUF_FORMAT;
    int wrapped_flags = supported_flags & ~PyBUF_WRITABLE;

    if (flags & ~supported_flags) {
        PyErr_SetString(PyExc_BufferError, "Unsupported flags for BufferProxy");
        view->obj = NULL;
        return -1;
    }

    if (self->ob_exports == 0) {
        if(-1 == PyObject_GetBuffer(self->wrapped, &(self->wrapped_buf),
                                    wrapped_flags)) {
            PyErr_SetString(PyExc_BufferError,
                            "Could not get buffer for wrapped (new protocol)");
            view->obj = NULL;
            return -1;
        }
    }

    if(-1 == PyBuffer_FillInfo(view, (PyObject*) self, self->wrapped_buf.buf,
                               self->wrapped_buf.len, 0, supported_flags)) {
        PyBuffer_Release(&(self->wrapped_buf));
        view->obj = NULL;
        return -1;
    }

    self->ob_exports += 1;
    return 0;
}

static void
BufferProxyObject_releasebuffer(BufferProxyObject *self, Py_buffer *view) {
    self->ob_exports -= 1;
    if (self->ob_exports == 0)
        PyBuffer_Release(&(self->wrapped_buf));
}

#if (PY_MAJOR_VERSION >= 3)
static PyBufferProcs BufferProxy_as_buffer = {
    (getbufferproc)BufferProxyObject_getbuffer,
    (releasebufferproc)BufferProxyObject_releasebuffer,
};
#else
static Py_ssize_t
BufferProxyObject_getwritebuf(BufferProxyObject *self, Py_ssize_t index,
                              const void **ptr) {
    if (index != 0) {
        PyErr_SetString(PyExc_TypeError,
                        "Accessing non-existent BufferProxy segment");
        return -1;
    }

    if (self->wrapped == NULL) {
        PyErr_SetString(PyExc_TypeError, "No wrapped object");
        return -1;
    }

    if (self->ob_old_exports == 0) {
        if (-1 == PyObject_AsReadBuffer(self->wrapped,
                                        (const void**) &(self->wrapped_ptr),
                                        &(self->wrapped_len))) {
            PyErr_SetString(PyExc_TypeError,
                            "Could not get buffer for wrapped (old protocol)");
            return -1;
        }
    }

    self->ob_old_exports += 1;
    *ptr = self->wrapped_ptr;
    return self->wrapped_len;
}

static Py_ssize_t
BufferProxyObject_getreadbuf(BufferProxyObject *self, Py_ssize_t index,
                             const void **ptr) {
    return BufferProxyObject_getwritebuf(self, index, ptr);
}

static Py_ssize_t
BufferProxyObject_getsegcount(BufferProxyObject *self, Py_ssize_t *lenp) {
    return self->wrapped->ob_type->tp_as_buffer->bf_getsegcount(self->wrapped,
                                                                lenp);
}

static Py_ssize_t
BufferProxyObject_getcharbuf(BufferProxyObject *self, Py_ssize_t index,
                             const char **ptr) {
    return BufferProxyObject_getwritebuf(self, index, (const void **) ptr);
}

static PyBufferProcs BufferProxy_as_buffer = {
    (readbufferproc)BufferProxyObject_getreadbuf,       /*bf_getreadbuffer*/
    (writebufferproc)BufferProxyObject_getwritebuf,     /*bf_getwritebuffer*/
    (segcountproc)BufferProxyObject_getsegcount,        /*bf_getsegcount*/
    (charbufferproc)BufferProxyObject_getcharbuf,       /*bf_getcharbuffer*/
    /* new buffer protocol */
    (getbufferproc)BufferProxyObject_getbuffer,         /*bf_getbuffer*/
    (releasebufferproc)BufferProxyObject_releasebuffer, /*bf_releasebuffer*/
};
#endif

static int
BufferProxy_init(BufferProxyObject *self, PyObject *args, PyObject *kwds) {
    if (!PyArg_ParseTuple(args, "O", &(self->wrapped))) {
        self->wrapped = NULL;
        return -1;
    }
    Py_INCREF(self->wrapped);
    self->ob_exports = 0;
#if (PY_MAJOR_VERSION < 3)
    self->ob_old_exports = 0;
#endif
    return 0;
}

static void
BufferProxy_dealloc(BufferProxyObject *self) {
    Py_XDECREF(self->wrapped);
}

static PyTypeObject BufferProxyType = {
#if (PY_MAJOR_VERSION < 3)
    PyObject_HEAD_INIT(NULL)
    0,                               /*ob_size*/
#else
    PyVarObject_HEAD_INIT(NULL, 0)
#endif
    "mviewbuf.BufferProxy",          /*tp_name*/
    sizeof(BufferProxyObject),       /*tp_basicsize*/
    0,                               /*tp_itemsize*/
    (destructor)BufferProxy_dealloc, /*tp_dealloc*/
    0,                               /*tp_print*/
    0,                               /*tp_getattr*/
    0,                               /*tp_setattr*/
    0,                               /*tp_compare*/
    0,                               /*tp_repr*/
    0,                               /*tp_as_number*/
    0,                               /*tp_as_sequence*/
    0,                               /*tp_as_mapping*/
    0,                               /*tp_hash */
    0,                               /*tp_call*/
    0,                               /*tp_str*/
    0,                               /*tp_getattro*/
    0,                               /*tp_setattro*/
    &BufferProxy_as_buffer,          /*tp_as_buffer*/
    (Py_TPFLAGS_DEFAULT
#if PY_MAJOR_VERSION < 3
     | Py_TPFLAGS_CHECKTYPES
#endif
#if (PY_VERSION_HEX >= 0x02060000) && (PY_VERSION_HEX < 0x03000000)
     | Py_TPFLAGS_HAVE_NEWBUFFER
#endif
     | Py_TPFLAGS_BASETYPE),         /*tp_flags*/
    "BufferProxy object",            /*tp_doc*/
    0,                               /*tp_traverse*/
    0,                               /*tp_clear*/
    0,                               /*tp_richcompare*/
    0,                               /*tp_weaklistoffset*/
    0,                               /*tp_iter*/
    0,                               /*tp_iternext*/
    0,                               /*tp_methods*/
    0,                               /*tp_members*/
    0,                               /*tp_getset*/
    0,                               /*tp_base*/
    0,                               /*tp_dict*/
    0,                               /*tp_descr_get*/
    0,                               /*tp_descr_set*/
    0,                               /*tp_dictoffset*/
    (initproc)BufferProxy_init,      /*tp_init*/
    0,                               /*tp_alloc*/
    0,                               /*tp_new*/
};

static int get_buffer(PyObject* obj, Py_buffer *buf)
{
    return PyObject_GetBuffer(obj, buf, PyBUF_WRITABLE|PyBUF_ND|PyBUF_STRIDES|PyBUF_FORMAT);
}

static void free_buffer(Py_buffer * buf)
{
    PyBuffer_Release(buf);
}

static PyObject*
memoryview_get_buffer(PyObject *self, PyObject *args){
    PyObject *obj = NULL;
    PyObject *ret = NULL;
    void * ptr = NULL;
    Py_ssize_t buflen;
    Py_buffer buf;

    if (!PyArg_ParseTuple(args, "O", &obj))
        return NULL;

    if (!get_buffer(obj, &buf)) { /* new buffer api */
        ret = PyLong_FromVoidPtr(buf.buf);
        free_buffer(&buf);
    } else { /* old buffer api */
        PyErr_Clear();
        if (-1 == PyObject_AsWriteBuffer(obj, &ptr, &buflen)) return NULL;
        ret = PyLong_FromVoidPtr(ptr);
    }
    return ret;
}

/**
 * Gets a half-open range [start, end) which contains the array data
 * Modified from numpy/core/src/multiarray/array_assign.c
 */
static PyObject*
get_extents(Py_ssize_t *shape, Py_ssize_t *strides, int ndim,
            Py_ssize_t itemsize, Py_ssize_t ptr)
{
    Py_ssize_t start, end;
    int idim;
    Py_ssize_t *dimensions = shape;
    PyObject *ret = NULL;

    if (ndim < 0 ){
        PyErr_SetString(PyExc_ValueError, "buffer ndim < 0");
        return NULL;
    }

    if (!dimensions) {
        if (ndim == 0) {
            start = end = ptr;
            end += itemsize;
            return Py_BuildValue("nn", start, end);
        }
        PyErr_SetString(PyExc_ValueError, "buffer shape is not defined");
        return NULL;
    }

    if (!strides) {
        PyErr_SetString(PyExc_ValueError, "buffer strides is not defined");
        return NULL;
    }

    /* Calculate with a closed range [start, end] */
    start = end = ptr;
    for (idim = 0; idim < ndim; ++idim) {
        Py_ssize_t stride = strides[idim], dim = dimensions[idim];
        /* If the array size is zero, return an empty range */
        if (dim == 0) {
            start = end = ptr;
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
        Py_ssize_t out_end = end + itemsize;

        ret = Py_BuildValue("nn", out_start, out_end);
    }

    return ret;
}

static PyObject*
memoryview_get_extents(PyObject *self, PyObject *args)
{
    PyObject *obj = NULL;
    PyObject *ret = NULL;
    Py_buffer b;
    const void * ptr = NULL;
    Py_ssize_t bufptr, buflen;
    if (!PyArg_ParseTuple(args, "O", &obj))
        return NULL;

    if (!get_buffer(obj, &b)) { /* new buffer api */
        ret = get_extents(b.shape, b.strides, b.ndim, b.itemsize,
                          (Py_ssize_t)b.buf);
        free_buffer(&b);
    } else { /* old buffer api */
        PyErr_Clear();
        if (-1 == PyObject_AsReadBuffer(obj, &ptr, &buflen)) return NULL;
        bufptr = (Py_ssize_t)ptr;
        ret = Py_BuildValue("nn", bufptr, bufptr + buflen);
    }
    return ret;
}

static PyObject*
memoryview_get_extents_info(PyObject *self, PyObject *args)
{
    int i;
    Py_ssize_t *shape_ary = NULL;
    Py_ssize_t *strides_ary = NULL;
    PyObject *shape_tuple = NULL;
    PyObject *strides_tuple = NULL;
    PyObject *shape = NULL, *strides = NULL;
    Py_ssize_t itemsize = 0;
    int ndim = 0;
    PyObject* res = NULL;

    if (!PyArg_ParseTuple(args, "OOin", &shape, &strides, &ndim, &itemsize))
        goto cleanup;

    if (ndim < 0) {
        PyErr_SetString(PyExc_ValueError, "ndim is negative");
        goto cleanup;
    }

    if (itemsize <= 0) {
        PyErr_SetString(PyExc_ValueError, "ndim <= 0");
        goto cleanup;
    }

    shape_ary = malloc(sizeof(Py_ssize_t) * ndim + 1);
    strides_ary = malloc(sizeof(Py_ssize_t) * ndim + 1);

    shape_tuple = PySequence_Fast(shape, "shape is not a sequence");
    if (!shape_tuple) goto cleanup;

    for (i = 0; i < ndim; ++i) {
        shape_ary[i] = PyNumber_AsSsize_t(
                           PySequence_Fast_GET_ITEM(shape_tuple, i),
                           PyExc_OverflowError);
    }

    strides_tuple = PySequence_Fast(strides, "strides is not a sequence");
    if (!strides_tuple) goto cleanup;

    for (i = 0; i < ndim; ++i) {
        strides_ary[i] = PyNumber_AsSsize_t(
                           PySequence_Fast_GET_ITEM(strides_tuple, i),
                           PyExc_OverflowError);
    }

    res = get_extents(shape_ary, strides_ary, ndim, itemsize, 0);
cleanup:
    free(shape_ary);
    free(strides_ary);
    Py_XDECREF(shape_tuple);
    Py_XDECREF(strides_tuple);
    return res;
}


/* new type to expose buffer interface */
typedef struct {
    PyObject_HEAD
    /* Type-specific fields go here. */
} MemAllocObject;


static int
get_bufinfo(PyObject *self, Py_ssize_t *psize, void **pptr)
{
    PyObject *buflen = NULL;
    PyObject *bufptr = NULL;
    Py_ssize_t size = 0;
    void* ptr = NULL;
    int ret = -1;

    buflen = PyObject_GetAttrString(self, "_buflen_");
    if (!buflen) goto cleanup;

    bufptr = PyObject_GetAttrString(self, "_bufptr_");
    if (!bufptr) goto cleanup;

    size = PyNumber_AsSsize_t(buflen, PyExc_OverflowError);
    if (size == -1 && PyErr_Occurred()) goto cleanup;
    else if (size < 0) {
        PyErr_SetString(PyExc_ValueError, "negative buffer size");
        goto cleanup;
    }

    ptr = PyLong_AsVoidPtr(PyNumber_Long(bufptr));
    if (PyErr_Occurred())
        goto cleanup;
    else if (!ptr) {
        PyErr_SetString(PyExc_ValueError, "null buffer pointer");
        goto cleanup;
    }

    *psize = size;
    *pptr = ptr;
    ret = 0;
cleanup:
    Py_XDECREF(buflen);
    Py_XDECREF(bufptr);
    return ret;
}


#if PY_MAJOR_VERSION >= 3


    static int
    MemAllocObject_getbuffer(PyObject *self, Py_buffer *view, int flags)
    {
        Py_ssize_t size = 0;
        void *ptr = 0;
        int readonly;

        if(-1 == get_bufinfo(self, &size, &ptr))
            return -1;

        readonly = (PyBUF_WRITABLE & flags) != PyBUF_WRITABLE;

        /* fill buffer */
        if (-1 == PyBuffer_FillInfo(view, self, (void*)ptr, size, readonly, flags))
            return -1;

        return 0;
    }

    static void
    MemAllocObject_releasebuffer(PyObject *self, Py_buffer *view)
    {
        /* Do nothing */
    }

    static PyBufferProcs MemAlloc_as_buffer = {
        MemAllocObject_getbuffer,
        MemAllocObject_releasebuffer,
    };
#else
    static int
    MemAllocObject_getbufferproc(PyObject *self, Py_buffer *view, int flags)
    {
        Py_ssize_t size = 0;
        void *ptr = 0;
        int readonly;

        if(-1 == get_bufinfo(self, &size, &ptr))
            return -1;

        readonly = (PyBUF_WRITABLE & flags) != PyBUF_WRITABLE;

        /* fill buffer */
        if (-1 == PyBuffer_FillInfo(view, self, (void*)ptr, size, readonly, flags))
            return -1;

        return 0;
    }

    static Py_ssize_t
    MemAllocObject_writebufferproc(PyObject *self, Py_ssize_t segment,
                                   void **ptrptr)
    {
        Py_ssize_t size = 0;
        if (segment != 0) {
            PyErr_SetString(PyExc_ValueError, "invalid segment");
            return -1;
        }

        if(-1 == get_bufinfo(self, &size, ptrptr))
            return -1;
        return size;
    }

    static Py_ssize_t
    MemAllocObject_readbufferproc(PyObject *self, Py_ssize_t segment,
                                  void **ptrptr)
    {
        return MemAllocObject_writebufferproc(self, segment, ptrptr);
    }


    static Py_ssize_t
    MemAllocObject_segcountproc(PyObject *self, Py_ssize_t *lenp)
    {
        void *ptr = 0;
        if (lenp){
            if(-1 == get_bufinfo(self, lenp, &ptr)) return 0;
        }
        return 1;
    }

    static Py_ssize_t
    MemAllocObject_charbufferproc(PyObject *self, Py_ssize_t segment,
                                  char **ptrptr)
    {
        return MemAllocObject_writebufferproc(self, segment, (void*)ptrptr);
    }

    static PyBufferProcs MemAlloc_as_buffer = {
        MemAllocObject_readbufferproc,      /*bf_getreadbuffer*/
        MemAllocObject_writebufferproc,     /*bf_getwritebuffer*/
        MemAllocObject_segcountproc,        /*bf_getsegcount*/
        MemAllocObject_charbufferproc,      /*bf_getcharbuffer*/
        /* new buffer protocol */
        MemAllocObject_getbufferproc,       /*bf_getbuffer*/
        NULL,                               /*bf_releasebuffer*/
    };
#endif

static PyTypeObject MemAllocType = {
#if PY_MAJOR_VERSION >= 3
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                                          /* ob_size */
#endif
    "mviewbuf.MemAlloc",                            /* tp_name */
    sizeof(MemAllocObject),                   /* tp_basicsize */
    0,                                          /* tp_itemsize */
    /* methods */
    0,                  /* tp_dealloc */
    0,                            /* tp_print */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
#if PY_MAJOR_VERSION >= 3
    0,                                          /* tp_reserved */
#else
    0,                                          /* tp_compare */
#endif
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    &MemAlloc_as_buffer,        /*tp_as_buffer*/
    (Py_TPFLAGS_DEFAULT
#if PY_MAJOR_VERSION < 3
     | Py_TPFLAGS_CHECKTYPES
#endif
#if (PY_VERSION_HEX >= 0x02060000) && (PY_VERSION_HEX < 0x03000000)
     | Py_TPFLAGS_HAVE_NEWBUFFER
#endif
     | Py_TPFLAGS_BASETYPE),                    /* tp_flags */
    0,                                          /* tp_doc */

    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    0,                                          /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    0,                                          /* tp_methods */
    0,                                          /* tp_members */
    0,                                          /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    0,                                          /* tp_init */
    0,                                          /* tp_alloc */
    0,                                          /* tp_new */
    0,                                          /* tp_free */
    0,                                          /* tp_is_gc */
    0,                                          /* tp_bases */
    0,                                          /* tp_mro */
    0,                                          /* tp_cache */
    0,                                          /* tp_subclasses */
    0,                                          /* tp_weaklist */
    0,                                          /* tp_del */
#if PY_VERSION_HEX >= 0x02060000
    0,                                          /* tp_version_tag */
#endif
};


static PyMethodDef core_methods[] = {
#define declmethod(func) { #func , ( PyCFunction )func , METH_VARARGS , NULL }
    declmethod(memoryview_get_buffer),
    declmethod(memoryview_get_extents),
    declmethod(memoryview_get_extents_info),
    { NULL },
#undef declmethod
};


MOD_INIT(mviewbuf) {
    PyObject *module;
    MOD_DEF(module, "mviewbuf", "No docs", core_methods)
    if (module == NULL)
        return MOD_ERROR_VAL;

    MemAllocType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&MemAllocType) < 0){
        return MOD_ERROR_VAL;
    }

    Py_INCREF(&MemAllocType);
    PyModule_AddObject(module, "MemAlloc", (PyObject*)&MemAllocType);

    BufferProxyType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&BufferProxyType) < 0){
        return MOD_ERROR_VAL;
    }

    Py_INCREF(&BufferProxyType);
    PyModule_AddObject(module, "BufferProxy", (PyObject*)&BufferProxyType);

    return MOD_SUCCESS_VAL(module);
}

