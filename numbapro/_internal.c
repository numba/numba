/*
 * Copyright (c) 2012 Continuum Analytics, Inc.
 * All Rights reserved.
 */

/* Python include */
#include "Python.h"
#include "numpy/ndarrayobject.h"
#include "numpy/ufuncobject.h"

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

#define ARRAY_C_ORDER 0x1

#define ARRAYS_ARE_CONTIG 0x2
#define ARRAYS_ARE_INNER_CONTIG 0x4
#define ARRAYS_ARE_MIXED_CONTIG 0x10
#define ARRAYS_ARE_STRIDED 0x20
#define ARRAYS_ARE_MIXED_STRIDED 0x40

/* Deallocate the PyArray_malloc calls */
static void
dyn_dealloc(PyUFuncObject *self)
{
    if (self->functions)
        PyArray_free(self->functions);
    if (self->types)
        PyArray_free(self->types);
    if (self->data)
        PyArray_free(self->data);
    Py_TYPE(self)->tp_base->tp_dealloc((PyObject *)self);
}


NPY_NO_EXPORT PyTypeObject PyDynUFunc_Type = {
#if defined(NPY_PY3K)
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                                          /* ob_size */
#endif
    "numbapro.dyn_ufunc",                       /* tp_name*/
    sizeof(PyUFuncObject),                      /* tp_basicsize*/
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
    0,                                          /* tp_call */
    0,                                          /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    0,                                          /* tp_flags */
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
                           char *name, char *doc, PyObject *object)
{
    PyObject *ufunc;

    ufunc = PyUFunc_FromFuncAndData(func, data, types, ntypes, nin, nout,
                                    identity, name, doc, 0);

    /* Kind of a gross-hack  */
    Py_TYPE(ufunc) = &PyDynUFunc_Type;

    /* Hold on to whatever object is passed in */
    Py_XINCREF(object);
    ((PyUFuncObject *)ufunc)->obj = object;

    return ufunc;
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
    int i, j;
    int custom_dtype = 0;
    PyUFuncGenericFunction *funcs;
    int *types;
    void **data;
    PyObject *ufunc;

    if (!PyArg_ParseTuple(args, "O!O!iiO|O", &PyList_Type, &func_list, &PyList_Type, &type_list, &nin, &nout, &data_list, &object)) {
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
        ufunc = PyDynUFunc_FromFuncAndData((PyUFuncGenericFunction*)funcs,data,(char*)char_types,nfuncs,nin,nout,PyUFunc_None,"test",(char*)"test",object);
    }
    else {
        ufunc = PyDynUFunc_FromFuncAndData(0,0,0,0,nin,nout,PyUFunc_None,"test",(char*)"test",object);
        PyUFunc_RegisterLoopForType((PyUFuncObject*)ufunc,custom_dtype,funcs[0],types,0);
        PyArray_free(funcs);
        PyArray_free(types);
        PyArray_free(data);
    }

    return ufunc;
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
    PyObject *ufunc;
    ufunc = PyUFunc_FromFuncAndDataAndSignature(func, data, types, ntypes, nin, nout,
                                    identity, name, doc, 0, signature);
    if(!ufunc)
        return ufunc;
    /* Kind of a gross-hack  */
    Py_TYPE(ufunc) = &PyDynUFunc_Type;

    /* Hold on to whatever object is passed in */
    Py_XINCREF(object);
    ((PyUFuncObject *)ufunc)->obj = object;

    return ufunc;
}

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
        ufunc = PyDynUFunc_FromFuncAndDataAndSignature((PyUFuncGenericFunction*)funcs,data,(char*)char_types,nfuncs,nin,nout,PyUFunc_None,"test",(char*)"test",signature,object);
    }
    else {
        ufunc = PyDynUFunc_FromFuncAndDataAndSignature(0,0,0,0,nin,nout,PyUFunc_None,"test",(char*)"test",signature,object);
        PyUFunc_RegisterLoopForType((PyUFuncObject*)ufunc,custom_dtype,funcs[0],types,0);
        PyArray_free(funcs);
        PyArray_free(types);
        PyArray_free(data);
    }

    return ufunc;
}

#define absval(val) (val < 0 ? -val : val)

/*
    Figure out the best memory access order for a given array, ignore broadcasting
*/
static char
get_best_order(PyArrayObject *array, int ndim)
{
    int i, j;

    npy_intp *shape = PyArray_DIMS(array);

    npy_intp c_stride = 0;
    npy_intp f_stride = 0;

    if (ndim == 1)
        return 'A'; /* short-circuit */

    for (i = ndim - 1; i >= 0; i--) {
        if (shape[i] != 1) {
            c_stride = PyArray_STRIDE(array, i);
            break;
        }
    }

    for (j = 0; j < ndim; j++) {
        if (shape[j] != 1) {
            f_stride = PyArray_STRIDE(array, i);
            break;
        }
    }

    if (i == j) {
        if (i > 0)
            return 'c';
        else
            return 'f';
    } else if (absval(c_stride) <= absval(f_stride)) {
        return 'C';
    } else {
        return 'F';
    }
}

/* Get the overall data order for a list of NumPy arrays for
 * element-wise traversal */
static PyObject *
get_arrays_ordering(PyObject *NPY_UNUSED(dummy), PyObject *args)
{
    int result_flags;

    int all_c_contig = 1;
    int all_f_contig = 1;
    int seen_c_contig = 0;
    int seen_f_contig = 0;
    int seen_c_ish = 0;
    int seen_f_ish = 0;

    int i;

    PyObject *arrays;
    int n_arrays;
    int broadcasting = 0;

    if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &arrays)) {
        return NULL;
    }

    n_arrays = PyList_GET_SIZE(arrays);
    if (!n_arrays) {
        PyErr_SetString(PyExc_ValueError, "Expected non-empty list of arrays");
        return NULL;
    }

    /* Count the data orders */
    for (i = 0; i < n_arrays; i++) {
        char order;

        PyArrayObject *array = (PyArrayObject *) PyList_GET_ITEM(arrays, i);
        int ndim = PyArray_NDIM(array);
        int c_contig = PyArray_ISCONTIGUOUS(array);
        int f_contig = PyArray_ISFORTRAN(array);

        if (!ndim)
            continue;

        if (c_contig) {
            order = 'C';
        } else if (f_contig) {
            order = 'F';
        } else {
            order = get_best_order(array, ndim);

            if (order == 'c' || order == 'f') {
                broadcasting++;
                order = toupper(order);
            }
        }

        if (order == 'C') {
            all_f_contig = 0;
            all_c_contig &= c_contig;
            seen_c_contig += c_contig;
            seen_c_ish++;
        } else {
            all_c_contig = 0;
            all_f_contig &= f_contig;
            seen_f_contig += f_contig;
            seen_f_ish++;
        }
    }

    if (all_c_contig || all_f_contig) {
        result_flags = ARRAYS_ARE_CONTIG | all_c_contig;
    } else if (broadcasting == n_arrays) {
        result_flags = ARRAYS_ARE_STRIDED | ARRAY_C_ORDER;
    } else if (seen_c_contig + seen_f_contig == n_arrays - broadcasting) {
        result_flags = ARRAYS_ARE_MIXED_CONTIG | (seen_c_ish > seen_f_ish);
    } else if (seen_c_ish && seen_f_ish) {
        result_flags = ARRAYS_ARE_MIXED_STRIDED | (seen_c_ish > seen_f_ish);
    } else {
        /*
           Check whether the operands are strided or inner contiguous.
           We check whether the stride in the first or last (F/C) dimension equals
           the itemsize, and we verify that no operand is broadcasting in the
           first or last (F/C) dimension (that they all have the same extent).
        */
        PyArrayObject *array = (PyArrayObject *) PyList_GET_ITEM(arrays, 0);
        npy_intp extent;

        if (seen_c_ish)
            extent = PyArray_DIM(array, PyArray_NDIM(array) - 1);
        else
            extent = PyArray_DIM(array, 0);

        /* Assume inner contiguous */
        result_flags = ARRAYS_ARE_INNER_CONTIG | !!seen_c_ish;
        for (i = 0; i < n_arrays; i++) {
            int dim = 0;
            array = (PyArrayObject *) PyList_GET_ITEM(arrays, i);
            if (seen_c_ish)
                dim = PyArray_NDIM(array) - 1;

            if (dim < 0)
                continue;

            if (PyArray_STRIDE(array, dim) != PyArray_ITEMSIZE(array) ||
                    PyArray_DIM(array, dim) != extent) {
                result_flags = ARRAYS_ARE_STRIDED | !!seen_c_ish;
                break;
            }
        }
    }

    return PyLong_FromLong(result_flags);
}

static int
add_array_order_constants(PyObject *module)
{
#define __err_if_neg(expr) if (expr < 0) return -1;
    __err_if_neg(PyModule_AddIntConstant(module, "ARRAY_C_ORDER", ARRAY_C_ORDER));
    __err_if_neg(PyModule_AddIntConstant(module, "ARRAYS_ARE_CONTIG", ARRAYS_ARE_CONTIG));
    __err_if_neg(PyModule_AddIntConstant(module, "ARRAYS_ARE_INNER_CONTIG", ARRAYS_ARE_INNER_CONTIG));
    __err_if_neg(PyModule_AddIntConstant(module, "ARRAYS_ARE_MIXED_CONTIG", ARRAYS_ARE_MIXED_CONTIG));
    __err_if_neg(PyModule_AddIntConstant(module, "ARRAYS_ARE_STRIDED", ARRAYS_ARE_STRIDED));
    __err_if_neg(PyModule_AddIntConstant(module, "ARRAYS_ARE_MIXED_STRIDED", ARRAYS_ARE_MIXED_STRIDED));
#undef __err_if_neg
    return 0;
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
    PyDynUFunc_Type.tp_base = &PyUFunc_Type;
    if (PyType_Ready(&PyDynUFunc_Type) < 0)
        return RETVAL;

    return RETVAL;
}

