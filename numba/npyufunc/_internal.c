/*
 * Copyright (c) 2012 Continuum Analytics, Inc.
 * All Rights reserved.
 */

#include "_internal.h"

/* A small object that handles deallocation of some of a PyUFunc's fields */
typedef struct {
    PyObject_HEAD
    /* Borrowed reference */
    PyUFuncObject *ufunc;
    /* Owned reference to ancilliary object */
    PyObject *object;
} PyUFuncCleaner;

PyTypeObject PyUFuncCleaner_Type;


static PyObject *
cleaner_new(PyUFuncObject *ufunc, PyObject *object)
{
    PyUFuncCleaner *obj = PyObject_New(PyUFuncCleaner, &PyUFuncCleaner_Type);
    if (obj != NULL) {
        obj->ufunc = ufunc;
        Py_XINCREF(object);
        obj->object = object;
    }
    return (PyObject *) obj;
}

/* Deallocate the PyArray_malloc calls */
static void
cleaner_dealloc(PyUFuncCleaner *self)
{
    PyUFuncObject *ufunc = self->ufunc;
    Py_XDECREF(self->object);
    if (ufunc->functions)
        PyArray_free(ufunc->functions);
    if (ufunc->types)
        PyArray_free(ufunc->types);
    if (ufunc->data)
        PyArray_free(ufunc->data);
    PyObject_Del(self);
}

PyTypeObject PyUFuncCleaner_Type = {
#if PY_MAJOR_VERSION >= 3
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                                          /* ob_size */
#endif
    "numba._UFuncCleaner",                      /* tp_name*/
    sizeof(PyUFuncCleaner),                     /* tp_basicsize*/
    0,                                          /* tp_itemsize */
    /* methods */
    (destructor) cleaner_dealloc,               /* tp_dealloc */
    0,                                          /* tp_print */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
#if defined(NPY_PY3K)
    0,                                          /* tp_reserved */
#else
    0,                                          /* tp_compare */
#endif
    0,                                          /* tp_repr */
    0,                                          /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    0,                                          /* tp_hash */
    0,                                          /* tp_call */
    0,                                          /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                         /* tp_flags */
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
    0,                                          /* tp_version_tag */
};

static PyMethodDef ext_methods[] = {
    {"fromfunc", (PyCFunction) ufunc_fromfunc, METH_VARARGS, NULL},
    { NULL }
};

/* Don't remove this marker, it is used for inserting licensing code */
/*MARK1*/

MOD_INIT(_internal)
{
    PyObject *m;

    /* Don't remove this marker, it is used for inserting licensing code */
    /*MARK2*/

    import_array();
    import_umath();

    MOD_DEF(m, "_internal", "No docs",
            ext_methods)

    if (m == NULL)
        return MOD_ERROR_VAL;

    if (PyModule_AddIntMacro(m, PyUFunc_One)
        || PyModule_AddIntMacro(m, PyUFunc_Zero)
        || PyModule_AddIntMacro(m, PyUFunc_None)
#if NPY_API_VERSION >= 0x00000007
        || PyModule_AddIntMacro(m, PyUFunc_ReorderableNone)
#endif
        )
        return MOD_ERROR_VAL;

    return MOD_SUCCESS_VAL(m);
}


#include "_ufunc.c"
