/* This file contains the base class implementation for all device arrays. The
 * base class is implemented in C so that computing typecodes for device arrays
 * can be implemented efficiently. */

#include "_pymodule.h"


/* Include _devicearray., but make sure we don't get the definitions intended
 * for consumers of the Device Array API.
 */
#define NUMBA_IN_DEVICEARRAY_CPP_
#include "_devicearray.h"

/* DeviceArray PyObject implementation. Note that adding more members here is
 * presently prohibited because mapped and managed arrays derive from both
 * DeviceArray and NumPy's ndarray, which is also a C extension class - the
 * layout of the object cannot be resolved if this class also has members beyond
 * PyObject_HEAD. */
class DeviceArray {
    PyObject_HEAD
};

/* Trivial traversal - DeviceArray instances own nothing. */
static int
DeviceArray_traverse(DeviceArray *self, visitproc visit, void *arg)
{
    return 0;
}

/* Trivial clear of all references - DeviceArray instances own nothing. */
static int
DeviceArray_clear(DeviceArray *self)
{
    return 0;
}

/* The _devicearray.DeviceArray type */
PyTypeObject DeviceArrayType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_devicearray.DeviceArray",                  /* tp_name */
    sizeof(DeviceArray),                         /* tp_basicsize */
    0,                                           /* tp_itemsize */
    0,                                           /* tp_dealloc */
    0,                                           /* tp_vectorcall_offset */
    0,                                           /* tp_getattr */
    0,                                           /* tp_setattr */
    0,                                           /* tp_as_async */
    0,                                           /* tp_repr */
    0,                                           /* tp_as_number */
    0,                                           /* tp_as_sequence */
    0,                                           /* tp_as_mapping */
    0,                                           /* tp_hash */
    0,                                           /* tp_call*/
    0,                                           /* tp_str*/
    0,                                           /* tp_getattro*/
    0,                                           /* tp_setattro*/
    0,                                           /* tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,
                                                 /* tp_flags*/
    "DeviceArray object",                        /* tp_doc */
    (traverseproc) DeviceArray_traverse,         /* tp_traverse */
    (inquiry) DeviceArray_clear,                 /* tp_clear */
    0,                                           /* tp_richcompare */
    0,                                           /* tp_weaklistoffset */
    0,                                           /* tp_iter */
    0,                                           /* tp_iternext */
    0,                                           /* tp_methods */
    0,                                           /* tp_members */
    0,                                           /* tp_getset */
    0,                                           /* tp_base */
    0,                                           /* tp_dict */
    0,                                           /* tp_descr_get */
    0,                                           /* tp_descr_set */
    0,                                           /* tp_dictoffset */
    0,                                           /* tp_init */
    0,                                           /* tp_alloc */
    0,                                           /* tp_new */
    0,                                           /* tp_free */
    0,                                           /* tp_is_gc */
    0,                                           /* tp_bases */
    0,                                           /* tp_mro */
    0,                                           /* tp_cache */
    0,                                           /* tp_subclasses */
    0,                                           /* tp_weaklist */
    0,                                           /* tp_del */
    0,                                           /* tp_version_tag */
    0,                                           /* tp_finalize */
/* The docs suggest Python 3.8 has no tp_vectorcall
 * https://github.com/python/cpython/blob/d917cfe4051d45b2b755c726c096ecfcc4869ceb/Doc/c-api/typeobj.rst?plain=1#L146
 * but the header has it:
 * https://github.com/python/cpython/blob/d917cfe4051d45b2b755c726c096ecfcc4869ceb/Include/cpython/object.h#L257
 */
    0,                                           /* tp_vectorcall */
#if (PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION == 8)
/* This is Python 3.8 only.
 * See: https://github.com/python/cpython/blob/3.8/Include/cpython/object.h
 * there's a tp_print preserved for backwards compatibility. xref:
 * https://github.com/python/cpython/blob/d917cfe4051d45b2b755c726c096ecfcc4869ceb/Include/cpython/object.h#L260
 */
    0,                                           /* tp_print */
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

/* CUDA device array C API */
static void *_DeviceArray_API[1] = {
    (void*)&DeviceArrayType
};

MOD_INIT(_devicearray) {
    PyObject *m = nullptr;
    PyObject *d = nullptr;
    PyObject *c_api = nullptr;
    int error = 0;

    MOD_DEF(m, "_devicearray", "No docs", NULL)
    if (m == NULL)
        goto error_occurred;

    c_api = PyCapsule_New((void *)_DeviceArray_API, "numba._devicearray._DEVICEARRAY_API", NULL);
    if (c_api == NULL)
        goto error_occurred;

    DeviceArrayType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&DeviceArrayType) < 0)
        goto error_occurred;

    Py_INCREF(&DeviceArrayType);
    error = PyModule_AddObject(m, "DeviceArray", (PyObject*)(&DeviceArrayType));
    if (error)
        goto error_occurred;

    d = PyModule_GetDict(m);
    if (d == NULL)
        goto error_occurred;

    error = PyDict_SetItemString(d, "_DEVICEARRAY_API", c_api);
    /* Decref and set c_api to NULL, Py_XDECREF in error_occurred will have no
     * effect. */
    Py_CLEAR(c_api);

    if (error)
        goto error_occurred;

    return MOD_SUCCESS_VAL(m);

error_occurred:
    Py_XDECREF(m);
    Py_XDECREF(c_api);
    Py_XDECREF((PyObject*)&DeviceArrayType);

    return MOD_ERROR_VAL;
}
