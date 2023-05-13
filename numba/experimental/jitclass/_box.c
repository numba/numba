/*
Implements jitclass Box type in python c-api level.
*/
#include "../../_pymodule.h"

typedef struct {
    PyObject_HEAD
    void *meminfoptr, *dataptr;
} BoxObject;


/* Store function defined in numba.core.runtime._nrt_python for use in box_dealloc.
 * It points to a function is code segment that does not need user deallocation
 * and does not disappear while the process is still running.
 */
static void (*MemInfo_release)(void*) = NULL;


/*
 * Box.__init__()
 * Takes no arguments.
 * meminfoptr and dataptr are set to NULL.
 */
static
int Box_init(BoxObject *self, PyObject *args, PyObject *kwds) {
    static char *keywords[] = {NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "", keywords))
    {
        return -1;
    }
    /* Initialize attributes to NULL */
    self->meminfoptr = NULL;
    self->dataptr = NULL;
    return 0;
}

/*
 * Box destructor
 * Release MemInfo pointed by meminfoptr.
 * Free the instance.
 */
static
void box_dealloc(BoxObject *box)
{
    if (box->meminfoptr) MemInfo_release((void*)box->meminfoptr);
    Py_TYPE(box)->tp_free((PyObject *) box);
}


static const char Box_doc[] = "A box for numba created jit-class instance";


static PyTypeObject BoxType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_box.Box",                               /* tp_name */
    sizeof(BoxObject),                        /* tp_basicsize */
    0,                                        /* tp_itemsize */
    (destructor)box_dealloc,                  /* tp_dealloc */
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
    0,                                        /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    Box_doc,                                  /* tp_doc */
    0,                                        /* tp_traverse */
    0,                                        /* tp_clear */
    0,                                        /* tp_richcompare */
    0,                                        /* tp_weaklistoffset */
    0,                                        /* tp_iter */
    0,                                        /* tp_iternext */
    0,                                        /* tp_methods */
    0,                                        /* tp_members */
    0,                                        /* tp_getset */
    0,                                        /* tp_base */
    0,                                        /* tp_dict */
    0,                                        /* tp_descr_get */
    0,                                        /* tp_descr_set */
    0,                                        /* tp_dictoffset */
    (initproc)Box_init,                       /* tp_init */
    0,                                        /* tp_alloc */
    PyType_GenericNew,                        /* tp_new */
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
/* The docs suggest Python 3.8 has no tp_vectorcall
 * https://github.com/python/cpython/blob/d917cfe4051d45b2b755c726c096ecfcc4869ceb/Doc/c-api/typeobj.rst?plain=1#L146
 * but the header has it:
 * https://github.com/python/cpython/blob/d917cfe4051d45b2b755c726c096ecfcc4869ceb/Include/cpython/object.h#L257
 */
    0,                                        /* tp_vectorcall */
#if (PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION == 8)
/* This is Python 3.8 only.
 * See: https://github.com/python/cpython/blob/3.8/Include/cpython/object.h
 * there's a tp_print preserved for backwards compatibility. xref:
 * https://github.com/python/cpython/blob/d917cfe4051d45b2b755c726c096ecfcc4869ceb/Include/cpython/object.h#L260
 */
    0,                                        /* tp_print */
#endif

/* WARNING: Do not remove this, only modify it! It is a version guard to
 * act as a reminder to update this struct on Python version update! */
#if (PY_MAJOR_VERSION == 3)
#if ! ((PY_MINOR_VERSION == 8) || (PY_MINOR_VERSION == 9) || (PY_MINOR_VERSION == 10) || (PY_MINOR_VERSION == 11))
#error "Python minor version is not supported."
#endif
#else
#error "Python major version is not supported."
#endif
/* END WARNING*/
};


/* Import MemInfo_Release from numba.core.runtime._nrt_python once for use in
 * Box_dealloc.
 */
static void *
import_meminfo_release(void) {
    PyObject *nrtmod = NULL;
    PyObject *helperdct = NULL;
    PyObject *mi_rel_fn = NULL;
    void *fnptr = NULL;
    /* from numba.core.runtime import _nrt_python */
    nrtmod = PyImport_ImportModule("numba.core.runtime._nrt_python");
    if (!nrtmod) goto cleanup;
    /* helperdct = _nrt_python.c_helpers */
    helperdct = PyObject_GetAttrString(nrtmod, "c_helpers");
    if (!helperdct) goto cleanup;
    /* helperdct['MemInfo_release'] */
    mi_rel_fn = PyDict_GetItemString(helperdct, "MemInfo_release");
    if (!mi_rel_fn) goto cleanup;
    fnptr = PyLong_AsVoidPtr(mi_rel_fn);

cleanup:
    Py_XDECREF(nrtmod);
    Py_XDECREF(helperdct);
    return fnptr;
}

/* Debug utils.
 * Get internal dataptr field from Box.
 */
static
PyObject* box_get_dataptr(PyObject *self, PyObject *args) {
    BoxObject *box;
    if (!PyArg_ParseTuple(args, "O!", &BoxType, (PyObject*)&box))
        return NULL;
    return PyLong_FromVoidPtr(box->dataptr);
}

/* Debug utils.
 * Get internal meminfoptr field from Box.
 */
static
PyObject* box_get_meminfoptr(PyObject *self, PyObject *args) {
    BoxObject *box;
    if (!PyArg_ParseTuple(args, "O!", &BoxType, (PyObject*)&box))
        return NULL;
    return PyLong_FromVoidPtr(box->meminfoptr);
}


static PyMethodDef ext_methods[] = {
#define declmethod(func) { #func , ( PyCFunction )func , METH_VARARGS , NULL }
    declmethod(box_get_dataptr),
    declmethod(box_get_meminfoptr),
    { NULL },
#undef declmethod
};


MOD_INIT(_box) {
    PyObject *m;

    MOD_DEF(m, "_box", "No docs", ext_methods)
    if (m == NULL)
        return MOD_ERROR_VAL;

    /* init BoxType */
    if (PyType_Ready(&BoxType))
        return MOD_ERROR_VAL;

    /* import and cache NRT_MemInfo_release function pointer */
    MemInfo_release = import_meminfo_release();
    if (!MemInfo_release) return MOD_ERROR_VAL;

    /* bind BoxType */
    Py_INCREF(&BoxType);
    PyModule_AddObject(m, "Box", (PyObject *) (&BoxType));

    /* bind address to direct access utils */;
    PyModule_AddObject(m, "box_meminfoptr_offset",
                       PyLong_FromSsize_t(offsetof(BoxObject, meminfoptr)));
    PyModule_AddObject(m, "box_dataptr_offset",
                       PyLong_FromSsize_t(offsetof(BoxObject, dataptr)));

    return MOD_SUCCESS_VAL(m);
}
