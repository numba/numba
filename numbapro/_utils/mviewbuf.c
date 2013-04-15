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

static PyObject*
memoryview_from_pointer(PyObject *self, PyObject *args){
    Py_ssize_t ptr, size;
    int readonly = 0; //writable
    int flags = PyBUF_CONTIG; // c-contiguous
    if (!PyArg_ParseTuple(args, "nn", &ptr, &size))
        return NULL;
    Py_buffer* buf = malloc(sizeof(Py_buffer));

    if(-1 == PyBuffer_FillInfo(buf, NULL, (void*)ptr, size, readonly, flags)){
        return NULL;
    }

    return PyMemoryView_FromBuffer(buf);
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


// new type to expose buffer interface
typedef struct {
    PyObject_HEAD
    /* Type-specific fields go here. */
} MemAllocObject;


static int
get_bufinfo(PyObject *self, Py_ssize_t *psize, void **pptr)
{
    PyObject *buflen = PyObject_GetAttrString(self, "_buflen_");
    if (!buflen) return -1;

    PyObject *bufptr = PyObject_GetAttrString(self, "_bufptr_");
    if (!bufptr) return -1;

    Py_ssize_t size = PyNumber_AsSsize_t(buflen, PyExc_OverflowError);
    if (size == -1 && PyErr_Occurred()) return -1;
    else if (size < 0) {
        PyErr_SetString(PyExc_ValueError, "negative buffer size");
        return -1;
    }

    void* ptr = (void*)PyNumber_AsSsize_t(bufptr, PyExc_OverflowError);
    if (PyErr_Occurred()) return -1;
    else if (!ptr) {
        PyErr_SetString(PyExc_ValueError, "null buffer pointer");
    }

    *psize = size;
    *pptr = ptr;

    return 0;
}

static int
MemAllocObject_getbufferproc(PyObject *self, Py_buffer *view, int flags)
{
    Py_ssize_t size = 0;
    void *ptr = 0;
    if(-1 == get_bufinfo(self, &size, &ptr))
        return -1;

    int readonly = (PyBUF_WRITABLE & flags) == PyBUF_WRITABLE;

    // fill buffer
    if (-1 == PyBuffer_FillInfo(view, self, (void*)ptr, size, readonly, flags))
        return -1;

    return 0;
}

static Py_ssize_t
MemAllocObject_writebufferproc(PyObject *self, Py_ssize_t segment,
                               void **ptrptr)
{
    if (segment != 0) {
        PyErr_SetString(PyExc_ValueError, "invalid segment");
        return -1;
    }

    Py_ssize_t size = 0;
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
    // new buffer protocol
    MemAllocObject_getbufferproc,       /*bf_getbuffer*/
    NULL,                               /*bf_releasebuffer*/
};

static PyTypeObject MemAllocType = {
#if defined(NPY_PY3K)
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
#if defined(NPY_PY3K)
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
#if !defined(NPY_PY3K)
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
    declmethod(memoryview_from_pointer),
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

        MemAllocType.tp_new = PyType_GenericNew;
        if (PyType_Ready(&MemAllocType) < 0)
            INITERROR;

        Py_INCREF(&MemAllocType);

        PyModule_AddObject(module, "MemAlloc", (PyObject*)&MemAllocType);
#if PY_MAJOR_VERSION >= 3

        return module;
#endif
    }

