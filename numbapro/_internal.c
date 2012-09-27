/*
 * Copyright (c) 2012 Continuum Analytics, Inc.
 * All Rights reserved.
 */


#include "miniutils.h"
#include "_internal.h"

PyObject *
PyDynUFunc_New(PyUFuncObject *ufunc, PyObject *minivect_dispatcher,
               PyObject *cuda_dispatcher, int use_cuda_gufunc)
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
    result->minivect_dispatcher = minivect_dispatcher;
    Py_XINCREF(minivect_dispatcher);
    result->cuda_dispatcher = cuda_dispatcher;
    Py_XINCREF(cuda_dispatcher);
    result->use_cuda_gufunc = use_cuda_gufunc;

    return (PyObject *) result;
}

/* Deallocate the PyArray_malloc calls */
static void
dyn_dealloc(PyDynUFuncObject *self)
{
    PyUFuncObject *ufunc = self->ufunc_original;
    Py_XDECREF(self->minivect_dispatcher);
    Py_XDECREF(self->cuda_dispatcher);

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
_dispatch_cuda(PyDynUFuncObject *self, PyObject *args, PyObject *kw)
{
    int i;
    /* Insert 'ufunc' ('self') as the first argument */
    PyObject *new_args = PyTuple_New(PyTuple_GET_SIZE(args) + 1);
    PyObject *result;

    if (!new_args)
        return NULL;

    Py_INCREF(self);
    PyTuple_SET_ITEM(new_args, 0, (PyObject *) self);
    for (i = 1; i < PyTuple_GET_SIZE(args) + 1; i++) {
        PyObject *obj = PyTuple_GET_ITEM(args, i - 1);
        Py_INCREF(obj);
        PyTuple_SET_ITEM(new_args, i, obj);
    }

    result = PyObject_Call(self->cuda_dispatcher, new_args, kw);
    Py_DECREF(new_args);
    return result;
}

static PyObject *
_dispatch_gufunc(PyDynUFuncObject *self, PyObject *args, PyObject *kw)
{
    int i;
    PyArrayObject *mps[NPY_MAXARGS] = { NULL };
    PyObject *retobj[NPY_MAXARGS];
    PyObject *result;
    PyUFuncObject *ufunc = &self->ufunc;

//    /* Call ufunc */
//    if (PyUFunc_GeneralizedFunction(ufunc, args, kw, mps) < 0) {
//        /* Free everything, take care of NPY_ARRAY_UPDATEIFCOPY */
//        for (i = 0; i < ufunc->nargs; i++) {
//            PyArray_XDECREF_ERR(mps[i]);
//        }
//        return NULL;
//
//    }

    /* Free the input references */
    for (i = 0; i < ufunc->nin; i++) {
        Py_XDECREF(mps[i]);
    }

    /* Build return value(s) */
    for (i = 0; i < ufunc->nout; i++) {
        int j = ufunc->nin+i;
        /* default behavior */
        retobj[i] = PyArray_Return(mps[j]);
    }

    if (ufunc->nout == 1) {
        result = retobj[0];
    } else {
        result = PyTuple_New(ufunc->nout);
        if (!result) {
            PyErr_NoMemory();
            goto fail;
        }
        for (i = 0; i < ufunc->nout; i++) {
            PyTuple_SET_ITEM(result, i, retobj[i]);
        }
    }

    return result;
fail:
    for (i = ufunc->nin; i < ufunc->nargs; i++) {
        Py_XDECREF(mps[i]);
    }
    return NULL;
}

static PyObject *
dyn_call(PyDynUFuncObject *self, PyObject *args, PyObject *kw)
{
    if (self->minivect_dispatcher) {
        return PyObject_Call(self->minivect_dispatcher, args, kw);
    } else if (self->cuda_dispatcher && !self->use_cuda_gufunc) {
        return _dispatch_cuda(self, args, kw);
    } else if (0 && self->ufunc.core_enabled && self->use_cuda_gufunc) {
        /* Generalized ufunc */
        /* return _dispatch_gufunc(self, args, kw); */
    }
    return PyDynUFunc_Type.tp_base->tp_call((PyObject *) self, args, kw);
}

/* NPY_NO_EXPORT */ PyTypeObject PyDynUFunc_Type = {
#if defined(NPY_PY3K)
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                                          /* ob_size */
#endif
    "numbapro.dyn_ufunc",                       /* tp_name*/
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

/*
#include "_ufunc.c"
#include "_gufunc.c"
*/

static PyMethodDef ext_methods[] = {

#ifdef IS_PY3K
    {"fromfunc", (PyCFunction) ufunc_fromfunc, METH_VARARGS, NULL},
    {"fromfuncsig", (PyCFunction) ufunc_fromfuncsig, METH_VARARGS, NULL},
    {"get_arrays_ordering", (PyCFunction) get_arrays_ordering, METH_VARARGS, NULL},
#else
    {"fromfunc", ufunc_fromfunc, METH_VARARGS, NULL},
    {"fromfuncsig", ufunc_fromfuncsig, METH_VARARGS, NULL},
    {"get_arrays_ordering", get_arrays_ordering, METH_VARARGS, NULL},
#endif
    { NULL }
};

/* Don't remove this marker, it is used for inserting licensing code */
/*MARK1*/

#if IS_PY3K

struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "_internal",
    NULL,
    -1,
    ext_methods,
    NULL, NULL, NULL, NULL
};
#endif

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

#ifdef IS_PY3K
#define RETVAL m
#define ERR_RETVAL NULL
PyObject *
PyInit__internal(void)
#else
#define RETVAL
#define ERR_RETVAL
PyMODINIT_FUNC
init_internal(void)
#endif
{
    PyObject *m;    //, *d; // unused

    /* Don't remove this marker, it is used for inserting licensing code */
    /*MARK2*/

    import_array();
    import_umath();
    init_ufunc();
    init_gufunc();

    if (PyErr_Occurred())
        return ERR_RETVAL;

#ifdef IS_PY3K
    m = PyModule_Create( &module_def );
#else
    m = Py_InitModule("_internal", ext_methods);
#endif

    if (add_array_order_constants(m) < 0)
        return ERR_RETVAL;

    if (add_ndarray_flags_constants(m) < 0)
        return ERR_RETVAL;

    /* Inherit the dynamic UFunc from UFunc */
    PyUFunc_Type.tp_flags |= Py_TPFLAGS_BASETYPE; /* Hack... */
    PyDynUFunc_Type.tp_base = &PyUFunc_Type;
    if (PyType_Ready(&PyDynUFunc_Type) < 0)
        return ERR_RETVAL;

    Py_INCREF(&PyDynUFunc_Type);
    if (PyModule_AddObject(m, "dyn_ufunc", (PyObject *) &PyDynUFunc_Type) < 0)
        return ERR_RETVAL;

    return RETVAL;
}
