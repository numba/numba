
/* Included by _internal.c */
#include "_internal.h"

static int
get_string(PyObject *obj, char **s, const char *type_error_message)
{
    *s = NULL;
    if (!PyString_Check(obj) && obj != Py_None) {
        PyErr_SetString(PyExc_TypeError, type_error_message);
        return -1;
    }
    if (obj != Py_None) {
        *s = PyString_AsString(obj);
        if (!*s)
            return -1;
    }
    return 0;
}


PyObject *
ufunc_fromfunc(PyObject *NPY_UNUSED(dummy), PyObject *args)
{
    int nin, nout;
    int nfuncs, ntypes, ndata;
    PyObject *func_list;
    PyObject *type_list;
    PyObject *data_list;
    PyObject *func_obj;
    PyObject *type_obj;
    PyObject *data_obj;
    PyObject *object; /* object to hold on to while ufunc is alive */
    PyObject *pyname, *pydoc;
    char *name = NULL, *doc = NULL;
    char *signature = NULL;
    int identity;

    int i, j;
    int custom_dtype = 0;
    PyUFuncGenericFunction *funcs;
    int *types;
    void **data;
    PyUFuncObject *ufunc;

    if (!PyArg_ParseTuple(args, "OOO!O!iiOOi|s",
                          &pyname, &pydoc,
                          &PyList_Type, &func_list,
                          &PyList_Type, &type_list,
                          &nin, &nout, &data_list,
                          &object, &identity, &signature)) {
        return NULL;
    }
    if (get_string(pyname, &name, "name should be str or None"))
        return NULL;
    if (get_string(pydoc, &doc, "doc should be str or None"))
        return NULL;
    /* Ensure the pointers to C strings stay alive until the ufunc dies. */
    object = PyTuple_Pack(3, object, pyname, pydoc);
    if (!object)
        return NULL;

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
        ufunc = (PyUFuncObject *) PyUFunc_FromFuncAndDataAndSignature(
            (PyUFuncGenericFunction*) funcs, data, (char*) char_types,
            nfuncs, nin, nout,
            identity,
            name, doc,
            0 /* check_return */, signature);
        if (!ufunc) {
            PyArray_free(funcs);
            PyArray_free(data);
            Py_DECREF(object);
            return NULL;
        }
        /* XXX funcs, char_types and data won't be free'ed when the ufunc dies */
    }
    else {
        ufunc = (PyUFuncObject *) PyUFunc_FromFuncAndDataAndSignature(
            0, 0, 0, 0,
            nin,
            nout,
            identity,
            name, doc,
            0 /* check_return */, signature);
        if (!ufunc) {
            PyArray_free(funcs);
            PyArray_free(data);
            PyArray_free(types);
            Py_DECREF(object);
            return NULL;
        }

        PyUFunc_RegisterLoopForType(ufunc,
                                    custom_dtype,
                                    funcs[0],
                                    types,
                                    0);
        PyArray_free(funcs);
        PyArray_free(types);
        PyArray_free(data);
        funcs = NULL;
        data = NULL;
    }

    /* Create the sentinel object to clean up dynamically-allocated fields
       when the ufunc is destroyed. */
    ufunc->obj = cleaner_new(ufunc, object);
    Py_DECREF(object);
    if (ufunc->obj == NULL) {
        PyArray_free(funcs);
        PyArray_free(data);
        Py_DECREF(ufunc);
        return NULL;
    }

    return (PyObject *) ufunc;
}
