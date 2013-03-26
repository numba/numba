/* Included by _internal.c */

#include "_internal.h"

MOD_INIT(ufunc) {

    PyObject *m;

    import_array();
    import_umath();

    MOD_DEF(m, "ufunc", "No docs", NULL)

    if (m == NULL)
        return MOD_ERROR_VAL;

    return MOD_SUCCESS_VAL(m);
}

static PyObject *
PyDynUFunc_FromFuncAndData(PyUFuncGenericFunction *func, void **data,
                           char *types, int ntypes,
                           int nin, int nout, int identity,
                           char *name, char *doc, PyObject *object,
                           PyObject *dispatcher)
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

    result = PyDynUFunc_New(ufunc, dispatcher);
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

PyObject *
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
    PyObject *dispatcher = NULL;

    int i, j;
    int custom_dtype = 0;
    PyUFuncGenericFunction *funcs;
    int *types;
    void **data;
    PyObject *ufunc;

    if (!PyArg_ParseTuple(args, "O!O!iiO|OO", &PyList_Type, &func_list,
                                               &PyList_Type, &type_list,
                                               &nin, &nout, &data_list,
                                               &dispatcher,
                                               &object)) {
        return NULL;
    }

    if (dispatcher == Py_None)
        dispatcher = NULL;

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
        if (!type_obj)
            return NULL;

        for (j = 0; j < (nin+nout); j++) {
            int dtype_num;
            PyObject *dtype_num_obj = PyList_GetItem(type_obj, j);
            if (!dtype_num_obj)
                return NULL;

            SENTRY_VALID_LONG(
                types[i*(nin+nout) + j] = PyLong_AsLong(dtype_num_obj)
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
        ufunc = PyDynUFunc_FromFuncAndData((PyUFuncGenericFunction*) funcs,
                                           data,
                                           (char*) char_types,
                                           nfuncs,
                                           nin,
                                           nout,
                                           PyUFunc_None,
                                           "test", (char*)
                                           "test", object,
                                           dispatcher);
    }
    else {
        ufunc = PyDynUFunc_FromFuncAndData(0, 0, 0, 0,
                                           nin,
                                           nout,
                                           PyUFunc_None,
                                           "test",
                                           (char*) "test",
                                           object,
                                           dispatcher);
        
        PyUFunc_RegisterLoopForType((PyUFuncObject*)ufunc,
                                    custom_dtype,
                                    funcs[0],
                                    types,
                                    0);
        PyArray_free(funcs);
        PyArray_free(types);
        PyArray_free(data);
    }

    return ufunc;
}
