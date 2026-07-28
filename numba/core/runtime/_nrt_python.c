/*
 * Definition of NRT functions for marshalling from / to Python objects.
 * This module is included by _nrt_pythonmod.c and by pycc-compiled modules.
 */

#include "../../_pymodule.h"

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
    NRT_Debug(nrt_debug_print("MemInfo_init self=%p raw_ptr=%p\n", self, raw_ptr));

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

static
PyObject*
MemInfo_get_external_allocator(MemInfoObject *self, void *closure) {
    void *p = NRT_MemInfo_external_allocator(self->meminfo);
    return PyLong_FromVoidPtr(p);
}

static
PyObject*
MemInfo_get_parent(MemInfoObject *self, void *closure) {
    void *p = NRT_MemInfo_parent(self->meminfo);
    if (p) {
        Py_INCREF(p);
        return (PyObject*)p;
    } else {
        Py_INCREF(Py_None);
        return Py_None;
    }
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
    {"external_allocator",
     (getter)MemInfo_get_external_allocator, NULL,
     "Get the external allocator",
     NULL},
    {"parent",
     (getter)MemInfo_get_parent, NULL,
     NULL},
    {NULL}  /* Sentinel */
};


static PyTypeObject MemInfoType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_nrt_python._MemInfo",                   /* tp_name */
    sizeof(MemInfoObject),                    /* tp_basicsize */
    0,                                        /* tp_itemsize */
    (destructor)MemInfo_dealloc,              /* tp_dealloc */
    0,                                        /* tp_vectorcall_offset */
    0,                                        /* tp_getattr */
    0,                                        /* tp_setattr */
    0,                                        /* tp_as_async */
    0,                                        /* tp_repr */
    0,                                        /* tp_as_number */
    0,                                        /* tp_as_sequence */
    0,                                        /* tp_as_mapping */
    0,                                        /* tp_hash */
    0,                                        /* tp_call */
    0,                                        /* tp_str */
    0,                                        /* tp_getattro */
    0,                                        /* tp_setattro */
    &MemInfo_bufferProcs,                     /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
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
    0,                                        /* tp_free */
    0,                                        /* tp_is_gc */
    0,                                        /* tp_bases */
    0,                                        /* tp_mro */
    0,                                        /* tp_cache */
    0,                                        /* tp_subclasses */
    0,                                        /* tp_weaklist */
    0,                                        /* tp_del */
    0,                                        /* tp_version_tag */
    0,                                        /* tp_finalize */
    0,                                        /* tp_vectorcall */
#if (PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION >= 12)
/* This was introduced first in 3.12
 * https://github.com/python/cpython/issues/91051
 */
    0,                                           /* tp_watched */
#endif
#if (PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION >= 13)
/* This was introduced in 3.13
 * https://github.com/python/cpython/pull/114900
 */
    0,                                           /* tp_versions_used */
#endif

/* WARNING: Do not remove this, only modify it! It is a version guard to
 * act as a reminder to update this struct on Python version update! */
#if (PY_MAJOR_VERSION == 3)
#if ! (NB_SUPPORTED_PYTHON_MINOR)
#error "Python minor version is not supported."
#endif
#else
#error "Python major version is not supported."
#endif
/* END WARNING*/
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


/* Initialization subroutines for modules including this source file */

static int
init_nrt_python_module(PyObject *module)
{
    MemInfoType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&MemInfoType))
        return -1;
    return 0;
}
