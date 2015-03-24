/*
 * Copyright (c) 2012 Continuum Analytics, Inc.
 * All Rights reserved.
 */

#include "_internal.h"
#include "Python.h"

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

/* ______________________________________________________________________
 * DUFunc: A call-time (hence dynamic) specializable ufunc.
 */

typedef struct {
    PyObject_HEAD
    PyUFuncObject * ufunc;
    PyObject * dispatcher;
    PyObject * keepalive;
} PyDUFuncObject;

static void
dufunc_dealloc(PyDUFuncObject *self)
{
    Py_XDECREF(self->ufunc);
    Py_XDECREF(self->dispatcher);
    Py_XDECREF(self->keepalive);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *
dufunc_repr(PyDUFuncObject *dufunc)
{
#if PY_MAJOR_VERSION >= 3
    return PyUnicode_FromFormat("<numba.dufunc '%s'>", dufunc->ufunc->name);
#else
    return PyString_FromFormat("<numba.dufunc '%s'>", dufunc->ufunc->name);
#endif
}

static PyObject *
dufunc_call(PyDUFuncObject *self, PyObject *args, PyObject *kws)
{
    PyObject *result = (PyObject *)NULL;

    result = PyUFunc_Type.tp_call((PyObject *)self->ufunc, args, kws);
    if (result == NULL) {
        /* TODO */
    }
    return result;
}

/* ____________________________________________________________
 * Shims to expose ufunc methods.
 */

static struct _ufunc_dispatch {
    PyCFunction ufunc_reduce;
    PyCFunction ufunc_accumulate;
    PyCFunction ufunc_reduceat;
    PyCFunction ufunc_outer;
    PyCFunction ufunc_at;
} ufunc_dispatch = {NULL, NULL, NULL, NULL, NULL};

static int
init_ufunc_dispatch(void)
{
    int result = 0;
    PyMethodDef * crnt = PyUFunc_Type.tp_methods;
    const char * crnt_name = NULL;
    for (; crnt->ml_name != NULL; crnt++) {
        crnt_name = crnt->ml_name;
        switch(crnt_name[0]) {
        case 'a':
            if (strncmp(crnt_name, "at", 3) == 0) {
                ufunc_dispatch.ufunc_at = crnt->ml_meth;
            } else if (strncmp(crnt_name, "accumulate", 11) == 0) {
                ufunc_dispatch.ufunc_accumulate = crnt->ml_meth;
            } else {
                result = -1;
            }
            break;
        case 'o':
            if (strncmp(crnt_name, "outer", 6) == 0) {
                ufunc_dispatch.ufunc_outer = crnt->ml_meth;
            } else {
                result = -1;
            }
            break;
        case 'r':
            if (strncmp(crnt_name, "reduce", 7) == 0) {
                ufunc_dispatch.ufunc_reduce = crnt->ml_meth;
            } else if (strncmp(crnt_name, "reduceat", 9) == 0) {
                ufunc_dispatch.ufunc_reduceat = crnt->ml_meth;
            } else {
                result = -1;
            }
            break;
        default:
            result = -1; /* Unknown method */
        }
        if (result < 0) break;
    }
    if (result == 0) {
        /* Sanity check. */
        result = ((ufunc_dispatch.ufunc_reduce != NULL) &&
                  (ufunc_dispatch.ufunc_accumulate != NULL) &&
                  (ufunc_dispatch.ufunc_reduceat != NULL) &&
                  (ufunc_dispatch.ufunc_outer != NULL) &&
                  (ufunc_dispatch.ufunc_at != NULL));
    }
    return result;
}

PyTypeObject PyDUFunc_Type = {
#if PY_MAJOR_VERSION >= 3
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                                          /* ob_size */
#endif
    "numba.DUFunc",                             /* tp_name*/
    sizeof(PyDUFuncObject),                     /* tp_basicsize*/
    0,                                          /* tp_itemsize */
    /* methods */
    (destructor) dufunc_dealloc,                /* tp_dealloc */
    0,                                          /* tp_print */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
#if defined(NPY_PY3K)
    0,                                          /* tp_reserved */
#else
    0,                                          /* tp_compare */
#endif
    (reprfunc) dufunc_repr,                     /* tp_repr */
    0,                                          /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    0,                                          /* tp_hash */
    (ternaryfunc) dufunc_call,                  /* tp_call */
    (reprfunc) dufunc_repr,                     /* tp_str */
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

static PyObject *
dufunc_fromfunc(PyObject *NPY_UNUSED(dummy), PyObject *args)
{
    PyDUFuncObject * dufunc = (PyDUFuncObject *)NULL;
    return (PyObject *)dufunc;
}

/* ______________________________________________________________________
 * Module initialization boilerplate follows.
 */

static PyMethodDef ext_methods[] = {
    {"fromfunc", (PyCFunction) ufunc_fromfunc, METH_VARARGS, NULL},
    {"dufunc", (PyCFunction) dufunc_fromfunc, METH_VARARGS, NULL},
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

    if (PyType_Ready(&PyUFuncCleaner_Type) < 0)
        return MOD_ERROR_VAL;

    PyDUFunc_Type.tp_new = PyType_GenericNew;
    if (init_ufunc_dispatch() != 0)
        return MOD_ERROR_VAL;
    if (PyType_Ready(&PyDUFunc_Type) < 0)
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
