/*
 * Copyright (c) 2012 Continuum Analytics, Inc.
 * All Rights reserved.
 */

#include "_internal.h"

PyObject *
PyDynUFunc_New(PyUFuncObject *ufunc, PyObject *dispatcher)
{
    /* PyDynUFuncObject *result = PyDynUFunc_Type.tp_base->tp_new(type, args, kw); */
    PyDynUFuncObject *result = PyObject_New(PyDynUFuncObject, &PyDynUFunc_Type);
    size_t ufunc_size;

    if (!result)
        return NULL;

    /* Gross hack, copy ufunc directly into our object, skipping
       the PyObject header. Hold on to the object to DECREF it
       when the dynufunc is deallocated. */
    ufunc_size = sizeof(PyUFuncObject) - offsetof(PyUFuncObject, nin);
    memcpy(&result->ufunc.nin, &ufunc->nin, ufunc_size);
    result->ufunc_original = ufunc;
    result->dispatcher = dispatcher;
    Py_XINCREF(dispatcher);
    return (PyObject *) result;
}

/* Deallocate the PyArray_malloc calls */
static void
dyn_dealloc(PyDynUFuncObject *self)
{
    PyUFuncObject *ufunc = self->ufunc_original;
    Py_XDECREF(self->dispatcher);

    if (ufunc->functions)
        PyArray_free(ufunc->functions);
    if (ufunc->types)
        PyArray_free(ufunc->types);
    if (ufunc->data)
        PyArray_free(ufunc->data);
    /* Py_TYPE(self)->tp_base->tp_dealloc((PyObject *)self); */
    Py_DECREF(ufunc);
}

static PyObject *
dyn_call(PyDynUFuncObject *self, PyObject *args, PyObject *kw)
{
    if (self->dispatcher) {
        return PyObject_Call(self->dispatcher, args, kw);
    } else{
        return PyDynUFunc_Type.tp_base->tp_call((PyObject *) self, args, kw);
    }
}

/* NPY_NO_EXPORT */ PyTypeObject PyDynUFunc_Type = {
#if PY_MAJOR_VERSION >= 3
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                                          /* ob_size */
#endif
    "numba.dyn_ufunc",                       /* tp_name*/
    sizeof(PyDynUFuncObject),                 /* tp_basicsize*/
    0,                                          /* tp_itemsize */
    /* methods */
    (destructor) dyn_dealloc,                                /* tp_dealloc */
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
    (ternaryfunc) dyn_call,                     /* tp_call */
    0,                                          /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE,     /* tp_flags */
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

static PyMethodDef ext_methods[] = {
    {"fromfunc", (PyCFunction) ufunc_fromfunc, METH_VARARGS, NULL},
    {"fromfuncsig", (PyCFunction) ufunc_fromfuncsig, METH_VARARGS, NULL},
    /*
    {"get_arrays_ordering", (PyCFunction) get_arrays_ordering, METH_VARARGS,
     NULL},
     */
    { NULL }
};

/* Don't remove this marker, it is used for inserting licensing code */
/*MARK1*/

/*
static int
add_ndarray_flags_constants(PyObject *module)
{
#define __err_if_neg(expr) if (expr < 0) return -1;
    __err_if_neg(PyModule_AddIntConstant(module, "NPY_WRITEABLE", NPY_WRITEABLE));
    __err_if_neg(PyModule_AddIntConstant(module, "NPY_ARRAY_ALIGNED", NPY_ARRAY_ALIGNED));
    __err_if_neg(PyModule_AddIntConstant(module, "NPY_ARRAY_OWNDATA", NPY_ARRAY_OWNDATA));
    __err_if_neg(PyModule_AddIntConstant(module, "NPY_ARRAY_C_CONTIGUOUS", NPY_ARRAY_C_CONTIGUOUS));
    __err_if_neg(PyModule_AddIntConstant(module, "NPY_ARRAY_F_CONTIGUOUS", NPY_ARRAY_F_CONTIGUOUS));
#undef __err_if_neg
    return 0;
}
*/

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
/*

    if (add_array_order_constants(m) < 0)
        return MOD_ERROR_VAL;

    if (add_ndarray_flags_constants(m) < 0)
        return MOD_ERROR_VAL;
*/

    /* Inherit the dynamic UFunc from UFunc */
    PyUFunc_Type.tp_flags |= Py_TPFLAGS_BASETYPE; /* Hack... */
    PyDynUFunc_Type.tp_base = &PyUFunc_Type;
    if (PyType_Ready(&PyDynUFunc_Type) < 0)
        return MOD_ERROR_VAL;

    Py_INCREF(&PyDynUFunc_Type);
    if (PyModule_AddObject(m, "dyn_ufunc", (PyObject *) &PyDynUFunc_Type) < 0)
        return MOD_ERROR_VAL;


    return MOD_SUCCESS_VAL(m);

}


#include "_ufunc.c"
#include "_gufunc.c"
