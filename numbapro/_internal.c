/*
 * Copyright (c) 2012 Continuum Analytics, Inc.
 * All Rights reserved.
 */

/* Python include */
#include "Python.h"
#include <structmember.h>

#include "numpy/ndarrayobject.h"
#include "numpy/ufuncobject.h"
#include "miniutils.h"

#if PY_MAJOR_VERSION >= 3
#define IS_PY3K
#endif

#define APPEND_(X, Y) X #Y
#define APPEND(X, Y) APPEND_(X, Y)
#define SENTRY_VALID_LONG(X) if( (X) == -1 ){                        \
    PyErr_SetString(PyExc_RuntimeError,                              \
                    APPEND("PyLong_AsLong overflow at ", __LINE__)); \
    return NULL;                                                     \
}

typedef struct {
    PyUFuncObject ufunc;
    PyUFuncObject *ufunc_original;
    PyObject *minivect_dispatcher;
    PyObject *cuda_dispatcher;
} PyDynUFuncObject;

extern PyTypeObject PyDynUFunc_Type;

static PyObject *
PyDynUFunc_New(PyUFuncObject *ufunc, PyObject *minivect_dispatcher,
               PyObject *cuda_dispatcher)
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
dyn_call(PyDynUFuncObject *self, PyObject *args, PyObject *kw)
{
    if (self->minivect_dispatcher) {
        return PyObject_Call(self->minivect_dispatcher, args, kw);
    } else if (self->cuda_dispatcher) {
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

static PyObject *
PyDynUFunc_FromFuncAndData(PyUFuncGenericFunction *func, void **data,
                           char *types, int ntypes,
                           int nin, int nout, int identity,
                           char *name, char *doc, PyObject *object,
                           PyObject *minivect_dispatcher,
                           PyObject *cuda_dispatcher)
{
    PyUFuncObject *ufunc = NULL;
    PyObject *result;

    ufunc = (PyUFuncObject *) PyUFunc_FromFuncAndData(
                        func, data, types, ntypes, nin, nout,
                        identity, name, doc, 0);
    if (!ufunc)
        goto err;

    /* Kind of a gross-hack  */
    /* Py_TYPE(ufunc) = &PyDynUFunc_Type; */

    result = PyDynUFunc_New(ufunc, minivect_dispatcher, cuda_dispatcher);
    if (!result)
        goto err;

    /* Hold on to whatever object is passed in */
    Py_XINCREF(object);
    ufunc->obj = object;

    return result;
err:
    Py_XDECREF(ufunc);
    return NULL;
}

//// Duplicate for FromFuncAndDataAndSignature
//// Need to refactor to reduce code duplication.

static PyObject *
PyDynUFunc_FromFuncAndDataAndSignature(PyUFuncGenericFunction *func, void **data,
                           char *types, int ntypes,
                           int nin, int nout, int identity,
                           char *name, char *doc, char *signature,
                           PyObject *object)
{
    PyUFuncObject *ufunc = NULL;
    PyObject *result;

    ufunc = (PyUFuncObject *) PyUFunc_FromFuncAndDataAndSignature(
                                    func, data, types, ntypes, nin, nout,
                                    identity, name, doc, 0, signature);
    if (!ufunc)
        return NULL;

    /* Kind of a gross-hack  */
    /* Py_TYPE(ufunc) = &PyDynUFunc_Type; */

    /* Hold on to whatever object is passed in */
    result = PyDynUFunc_New(ufunc, NULL, NULL);
    if (!result)
        goto err;

    /* Hold on to whatever object is passed in */
    Py_XINCREF(object);
    ufunc->obj = object;

    return result;
err:
    Py_XDECREF(ufunc);
    return NULL;
}



static PyObject *
ufunc_fromfunc(PyObject *NPY_UNUSED(dummy), PyObject *args) {

    // unsigned long func_address; // unused
    int nin, nout;
    int nfuncs, ntypes, ndata;
    PyObject *func_list;
    PyObject *type_list;
    PyObject *data_list;
    PyObject *func_obj;
    PyObject *type_obj;
    PyObject *data_obj;
    PyObject *object=NULL; /* object to hold on to while ufunc is alive */
    PyObject *minivect_dispatcher = NULL;
    PyObject *cuda_dispatcher = NULL;

    int i, j;
    int custom_dtype = 0;
    PyUFuncGenericFunction *funcs;
    int *types;
    void **data;
    PyObject *ufunc;

    if (!PyArg_ParseTuple(args, "O!O!iiO|OOO", &PyList_Type, &func_list,
                                               &PyList_Type, &type_list,
                                               &nin, &nout, &data_list,
                                               &minivect_dispatcher,
                                               &cuda_dispatcher,
                                               &object)) {
        return NULL;
    }

    if (minivect_dispatcher == Py_None)
        minivect_dispatcher = NULL;

    if (cuda_dispatcher == Py_None)
        cuda_dispatcher = NULL;

    nfuncs = PyList_Size(func_list);

    ntypes = PyList_Size(type_list);
    if (ntypes != nfuncs) {
        PyErr_SetString(PyExc_TypeError, "length of types list must be same as length of function pointer list");
        return NULL;
    }

    ndata = PyList_Size(data_list);
    if (ndata != nfuncs) {
        PyErr_SetString(PyExc_TypeError, "length of data pointer list must be same as length of function pointer list");
        return NULL;
    }

    funcs = PyArray_malloc(nfuncs * sizeof(PyUFuncGenericFunction));
    if (funcs == NULL) {
        return NULL;
    }

    /* build function pointer array */
    for (i = 0; i < nfuncs; i++) {
        func_obj = PyList_GetItem(func_list, i);
        /* Function pointers are passed in as long objects.
           Is there a better way to do this? */
        if (PyLong_Check(func_obj)) {
            funcs[i] = (PyUFuncGenericFunction)PyLong_AsVoidPtr(func_obj);
        }
        else {
            PyErr_SetString(PyExc_TypeError, "function pointer must be long object, or None");
            return NULL;
        }
    }

    types = PyArray_malloc(nfuncs * (nin+nout) * sizeof(int));
    if (types == NULL) {
        return NULL;
    }

    /* build function signatures array */
    for (i = 0; i < nfuncs; i++) {
        type_obj = PyList_GetItem(type_list, i);

        for (j = 0; j < (nin+nout); j++) {
            int dtype_num;

            SENTRY_VALID_LONG(
                types[i*(nin+nout) + j] = PyLong_AsLong(PyList_GetItem(type_obj, j))
            );

            dtype_num = PyLong_AsLong(PyList_GetItem(type_obj, j));

            SENTRY_VALID_LONG(dtype_num);

            if (dtype_num >= NPY_USERDEF) {
                custom_dtype = dtype_num;
            }
        }
    }

    data = PyArray_malloc(nfuncs * sizeof(void *));
    if (data == NULL) {
        return NULL;
    }

    /* build function data pointers array */
    for (i = 0; i < nfuncs; i++) {
        if (PyList_Check(data_list)) {
            data_obj = PyList_GetItem(data_list, i);
            if (PyLong_Check(data_obj)) {
                data[i] = PyLong_AsVoidPtr(data_obj);
            }
            else if (data_obj == Py_None) {
                data[i] = NULL;
            }
            else {
                PyErr_SetString(PyExc_TypeError, "data pointer must be long object, or None");
                return NULL;
            }
        }
        else if (data_list == Py_None) {
            data[i] = NULL;
        }
        else {
            PyErr_SetString(PyExc_TypeError, "data pointers argument must be a list of void pointers, or None");
            return NULL;
        }
    }

    if (!custom_dtype) {
        char *char_types = PyArray_malloc(nfuncs * (nin+nout) * sizeof(char));
        for (i = 0; i < nfuncs; i++) {
            for (j = 0; j < (nin+nout); j++) {
                char_types[i*(nin+nout) + j] = (char)types[i*(nin+nout) + j];
            }
        }
        PyArray_free(types);
        ufunc = PyDynUFunc_FromFuncAndData((PyUFuncGenericFunction*) funcs, data, (char*) char_types, nfuncs,
                                           nin, nout, PyUFunc_None, "test", (char*) "test", object,
                                           minivect_dispatcher, cuda_dispatcher);
    }
    else {
        ufunc = PyDynUFunc_FromFuncAndData(0,0,0,0, nin, nout, PyUFunc_None,
                                           "test", (char*) "test", object,
                                           minivect_dispatcher, cuda_dispatcher);
        PyUFunc_RegisterLoopForType((PyUFuncObject*)ufunc,custom_dtype,funcs[0],types,0);
        PyArray_free(funcs);
        PyArray_free(types);
        PyArray_free(data);
    }

    return ufunc;
}

/*
    Create a generalized ufunc
*/
static PyObject *
ufunc_fromfuncsig(PyObject *NPY_UNUSED(dummy), PyObject *args) {

    // unsigned long func_address; // unused
    int nin, nout;
    int nfuncs, ntypes, ndata;
    PyObject *func_list;
    PyObject *type_list;
    PyObject *data_list;
    PyObject *func_obj;
    PyObject *type_obj;
    PyObject *data_obj;
    PyObject *object=NULL; /* object to hold on to while ufunc is alive */
    int i, j;
    int custom_dtype = 0;
    PyUFuncGenericFunction *funcs;
    int *types;
    void **data;
    PyObject *ufunc;
    char * signature;

    if (!PyArg_ParseTuple(args, "O!O!iiOs|O", &PyList_Type, &func_list, &PyList_Type, &type_list, &nin, &nout, &data_list, &signature, &object)) {
        return NULL;
    }

    nfuncs = PyList_Size(func_list);

    ntypes = PyList_Size(type_list);
    if (ntypes != nfuncs) {
        PyErr_SetString(PyExc_TypeError, "length of types list must be same as length of function pointer list");
        return NULL;
    }

    ndata = PyList_Size(data_list);
    if (ndata != nfuncs) {
        PyErr_SetString(PyExc_TypeError, "length of data pointer list must be same as length of function pointer list");
        return NULL;
    }

    funcs = PyArray_malloc(nfuncs * sizeof(PyUFuncGenericFunction));
    if (funcs == NULL) {
        return NULL;
    }

    /* build function pointer array */
    for (i = 0; i < nfuncs; i++) {
        func_obj = PyList_GetItem(func_list, i);
        /* Function pointers are passed in as long objects.
           Is there a better way to do this? */
        if (PyLong_Check(func_obj)) {
            funcs[i] = (PyUFuncGenericFunction)PyLong_AsVoidPtr(func_obj);
        }
        else {
            PyErr_SetString(PyExc_TypeError, "function pointer must be long object, or None");
            return NULL;
        }
    }

    types = PyArray_malloc(nfuncs * (nin+nout) * sizeof(int));
    if (types == NULL) {
        return NULL;
    }

    /* build function signatures array */
    for (i = 0; i < nfuncs; i++) {
        type_obj = PyList_GetItem(type_list, i);

        for (j = 0; j < (nin+nout); j++) {
            int dtype_num;

            SENTRY_VALID_LONG(
                types[i*(nin+nout) + j] = PyLong_AsLong(PyList_GetItem(type_obj, j))
            );

            dtype_num = PyLong_AsLong(PyList_GetItem(type_obj, j));

            SENTRY_VALID_LONG(dtype_num);

            if (dtype_num >= NPY_USERDEF) {
                custom_dtype = dtype_num;
            }
        }
    }

    data = PyArray_malloc(nfuncs * sizeof(void *));
    if (data == NULL) {
        return NULL;
    }

    /* build function data pointers array */
    for (i = 0; i < nfuncs; i++) {
        if (PyList_Check(data_list)) {
            data_obj = PyList_GetItem(data_list, i);
            if (PyLong_Check(data_obj)) {
                data[i] = PyLong_AsVoidPtr(data_obj);
            }
            else if (data_obj == Py_None) {
                data[i] = NULL;
            }
            else {
                PyErr_SetString(PyExc_TypeError, "data pointer must be long object, or None");
                return NULL;
            }
        }
        else if (data_list == Py_None) {
            data[i] = NULL;
        }
        else {
            PyErr_SetString(PyExc_TypeError, "data pointers argument must be a list of void pointers, or None");
            return NULL;
        }
    }

    if (!custom_dtype) {
        char *char_types = PyArray_malloc(nfuncs * (nin+nout) * sizeof(char));
        for (i = 0; i < nfuncs; i++) {
            for (j = 0; j < (nin+nout); j++) {
                char_types[i*(nin+nout) + j] = (char)types[i*(nin+nout) + j];
            }
        }
        PyArray_free(types);
        ufunc = PyDynUFunc_FromFuncAndDataAndSignature(
                (PyUFuncGenericFunction*)funcs, data, (char*) char_types, nfuncs,
                nin, nout, PyUFunc_None, "test", (char*)"test", signature, object);
    }
    else {
        ufunc = PyDynUFunc_FromFuncAndDataAndSignature(
            0,0,0,0,nin,nout,PyUFunc_None,"test",(char*)"test",signature,object);
        PyUFunc_RegisterLoopForType((PyUFuncObject*)ufunc,custom_dtype,funcs[0],types,0);
        PyArray_free(funcs);
        PyArray_free(types);
        PyArray_free(data);
    }

    return ufunc;
}

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

#ifdef IS_PY3K
    m = PyModule_Create( &module_def );
#else
    m = Py_InitModule("_internal", ext_methods);
#endif

    if (add_array_order_constants(m) < 0)
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

