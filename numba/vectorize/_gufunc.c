/* Included by _internal.c */

#include "_internal.h"

#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"
#include "numpy/arrayscalars.h"
#include "numpy/ndarraytypes.h"

MOD_INIT(gufunc) {

    PyObject *m;

    import_array();
    import_umath();

    MOD_DEF(m, "gufunc", "No docs", NULL)

    if (m == NULL)
        return MOD_ERROR_VAL;

    return MOD_SUCCESS_VAL(m);
}


#define NPY_UF_DBG_PRINT(string) puts(string)
#define NPY_UF_DBG_PRINT1(string, arg) printf(string, arg)

/* Duplicate for FromFuncAndDataAndSignature
   Need to refactor to reduce code duplication. */
static PyObject *
PyDynUFunc_FromFuncAndDataAndSignature(PyUFuncGenericFunction *func,
                                       void **data,
                                       char *types,
                                       int ntypes,
                                       int nin,
                                       int nout,
                                       int identity,
                                       char *name,
                                       char *doc,
                                       char *signature,
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
    result = PyDynUFunc_New(ufunc, NULL);
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

/*
    Create a generalized ufunc
*/
PyObject *
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
    PyObject *object = NULL; /* object to hold on to while ufunc is alive */

    int i, j;
    int custom_dtype = 0;
    PyUFuncGenericFunction *funcs;
    int *types;
    void **data;
    PyObject *ufunc;
    char * signature;

    if (!PyArg_ParseTuple(args, "O!O!iiOs|O", &PyList_Type, &func_list,
                                               &PyList_Type, &type_list,
                                               &nin, &nout, &data_list,
                                               &signature,
                                               &object)) {
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
                types[i*(nin+nout) + j] = PyLong_AsLong(
                                                PyList_GetItem(type_obj, j))
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
                                         (PyUFuncGenericFunction*)funcs,
                                         data,
                                         (char*) char_types,
                                         nfuncs,
                                         nin,
                                         nout,
                                         PyUFunc_None,
                                         "test",
                                         (char*)"test",
                                         signature,
                                         object);
    }
    else {
        ufunc = PyDynUFunc_FromFuncAndDataAndSignature(0, 0, 0, 0,
                                                       nin,
                                                       nout,
                                                       PyUFunc_None,
                                                       "test",
                                                       (char*)"test",
                                                       signature,
                                                       object);
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

