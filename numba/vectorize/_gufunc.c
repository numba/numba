/* Included by _internal.c */

#include "_internal.h"

#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"
#include "numpy/arrayscalars.h"
#include "numpy/ndarraytypes.h"

INIT(init_gufunc) {
    import_array();
    import_umath();
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

///*
// * Extracts some values from the global pyvals tuple.
// * ref - should hold the global tuple
// * name - is the name of the ufunc (ufuncobj->name)
// * bufsize - receives the buffer size to use
// * errmask - receives the bitmask for error handling
// * errobj - receives the python object to call with the error,
// *          if an error handling method is 'call'
// */
//static int
//_extract_pyvals(PyObject *ref, char *name, int *bufsize,
//                int *errmask, PyObject **errobj)
//{
//    PyObject *retval;
//
//    *errobj = NULL;
//    if (!PyList_Check(ref) || (PyList_GET_SIZE(ref)!=3)) {
//        PyErr_Format(PyExc_TypeError,
//                "%s must be a length 3 list.", UFUNC_PYVALS_NAME);
//        return -1;
//    }
//
//    *bufsize = PyInt_AsLong(PyList_GET_ITEM(ref, 0));
//    if ((*bufsize == -1) && PyErr_Occurred()) {
//        return -1;
//    }
//    if ((*bufsize < NPY_MIN_BUFSIZE) ||
//            (*bufsize > NPY_MAX_BUFSIZE) ||
//            (*bufsize % 16 != 0)) {
//        PyErr_Format(PyExc_ValueError,
//                "buffer size (%d) is not in range "
//                "(%"NPY_INTP_FMT" - %"NPY_INTP_FMT") or not a multiple of 16",
//                *bufsize, (npy_intp) NPY_MIN_BUFSIZE,
//                (npy_intp) NPY_MAX_BUFSIZE);
//        return -1;
//    }
//
//    *errmask = PyInt_AsLong(PyList_GET_ITEM(ref, 1));
//    if (*errmask < 0) {
//        if (PyErr_Occurred()) {
//            return -1;
//        }
//        PyErr_Format(PyExc_ValueError,
//                     "invalid error mask (%d)",
//                     *errmask);
//        return -1;
//    }
//
//    retval = PyList_GET_ITEM(ref, 2);
//    if (retval != Py_None && !PyCallable_Check(retval)) {
//        PyObject *temp;
//        temp = PyObject_GetAttrString(retval, "write");
//        if (temp == NULL || !PyCallable_Check(temp)) {
//            PyErr_SetString(PyExc_TypeError,
//                            "python object must be callable or have " \
//                            "a callable write method");
//            Py_XDECREF(temp);
//            return -1;
//        }
//        Py_DECREF(temp);
//    }
//
//    *errobj = Py_BuildValue("NO", PyBytes_FromString(name), retval);
//    if (*errobj == NULL) {
//        return -1;
//    }
//    return 0;
//}
//
///********* GENERIC UFUNC USING ITERATOR *********/
//
///*
// * Parses the positional and keyword arguments for a generic ufunc call.
// *
// * Note that if an error is returned, the caller must free the
// * non-zero references in out_op.  This
// * function does not do its own clean-up.
// */
//static int get_ufunc_arguments(PyUFuncObject *ufunc,
//                PyObject *args, PyObject *kwds,
//                PyArrayObject **out_op,
//                NPY_ORDER *out_order,
//                NPY_CASTING *out_casting,
//                PyObject **out_extobj,
//                PyObject **out_typetup,
//                int *out_subok,
//                PyArrayObject **out_wheremask)
//{
//    int i, nargs, nin = ufunc->nin, nout = ufunc->nout;
//    PyObject *obj, *context;
//    PyObject *str_key_obj = NULL;
//    char *ufunc_name;
//
//    int any_flexible = 0, any_object = 0;
//
//    ufunc_name = ufunc->name ? ufunc->name : "<unnamed ufunc>";
//
//    *out_extobj = NULL;
//    *out_typetup = NULL;
//    if (out_wheremask != NULL) {
//        *out_wheremask = NULL;
//    }
//
//    /* Check number of arguments */
//    nargs = PyTuple_Size(args);
//    if ((nargs < nin) || (nargs > ufunc->nargs)) {
//        PyErr_SetString(PyExc_ValueError, "invalid number of arguments");
//        return -1;
//    }
//
//    /* Get input arguments */
//    for (i = 0; i < nin; ++i) {
//        obj = PyTuple_GET_ITEM(args, i);
//        if (!PyArray_Check(obj) && !PyArray_IsScalar(obj, Generic)) {
//            /*
//             * TODO: There should be a comment here explaining what
//             *       context does.
//             */
//            context = Py_BuildValue("OOi", ufunc, args, i);
//            if (context == NULL) {
//                return -1;
//            }
//        }
//        else {
//            context = NULL;
//        }
//        out_op[i] = (PyArrayObject *)PyArray_FromAny(obj,
//                                    NULL, 0, 0, 0, context);
//        Py_XDECREF(context);
//        if (out_op[i] == NULL) {
//            return -1;
//        }
//        if (!any_flexible &&
//                PyTypeNum_ISFLEXIBLE(PyArray_DESCR(out_op[i])->type_num)) {
//            any_flexible = 1;
//        }
//        if (!any_object &&
//                PyTypeNum_ISOBJECT(PyArray_DESCR(out_op[i])->type_num)) {
//            any_object = 1;
//        }
//    }
//
//    /*
//     * Indicate not implemented if there are flexible objects (structured
//     * type or string) but no object types.
//     *
//     * Not sure - adding this increased to 246 errors, 150 failures.
//     */
//    if (any_flexible && !any_object) {
//        return -2;
//
//    }
//
//    /* Get positional output arguments */
//    for (i = nin; i < nargs; ++i) {
//        obj = PyTuple_GET_ITEM(args, i);
//        /* Translate None to NULL */
//        if (obj == Py_None) {
//            continue;
//        }
//        /* If it's an array, can use it */
//        if (PyArray_Check(obj)) {
//            if (PyArray_FailUnlessWriteable((PyArrayObject *)obj,
//                                            "output array") < 0) {
//                return -1;
//            }
//            Py_INCREF(obj);
//            out_op[i] = (PyArrayObject *)obj;
//        }
//        else {
//            PyErr_SetString(PyExc_TypeError,
//                            "return arrays must be "
//                            "of ArrayType");
//            return -1;
//        }
//    }
//
//    /*
//     * Get keyword output and other arguments.
//     * Raise an error if anything else is present in the
//     * keyword dictionary.
//     */
//    if (kwds != NULL) {
//        PyObject *key, *value;
//        Py_ssize_t pos = 0;
//        while (PyDict_Next(kwds, &pos, &key, &value)) {
//            Py_ssize_t length = 0;
//            char *str = NULL;
//            int bad_arg = 1;
//
//#if defined(NPY_PY3K)
//            Py_XDECREF(str_key_obj);
//            str_key_obj = PyUnicode_AsASCIIString(key);
//            if (str_key_obj != NULL) {
//                key = str_key_obj;
//            }
//#endif
//
//            if (PyBytes_AsStringAndSize(key, &str, &length) == -1) {
//                PyErr_SetString(PyExc_TypeError, "invalid keyword argument");
//                goto fail;
//            }
//
//            switch (str[0]) {
//                case 'c':
//                    /* Provides a policy for allowed casting */
//                    if (strncmp(str,"casting",7) == 0) {
//                        if (!PyArray_CastingConverter(value, out_casting)) {
//                            goto fail;
//                        }
//                        bad_arg = 0;
//                    }
//                    break;
//                case 'd':
//                    /* Another way to specify 'sig' */
//                    if (strncmp(str,"dtype",5) == 0) {
//                        /* Allow this parameter to be None */
//                        PyArray_Descr *dtype;
//                        if (!PyArray_DescrConverter2(value, &dtype)) {
//                            goto fail;
//                        }
//                        if (dtype != NULL) {
//                            if (*out_typetup != NULL) {
//                                PyErr_SetString(PyExc_RuntimeError,
//                                    "cannot specify both 'sig' and 'dtype'");
//                                goto fail;
//                            }
//                            *out_typetup = Py_BuildValue("(N)", dtype);
//                        }
//                        bad_arg = 0;
//                    }
//                    break;
//                case 'e':
//                    /*
//                     * Overrides the global parameters buffer size,
//                     * error mask, and error object
//                     */
//                    if (strncmp(str,"extobj",6) == 0) {
//                        *out_extobj = value;
//                        bad_arg = 0;
//                    }
//                    break;
//                case 'o':
//                    /* First output may be specified as a keyword parameter */
//                    if (strncmp(str,"out",3) == 0) {
//                        if (out_op[nin] != NULL) {
//                            PyErr_SetString(PyExc_ValueError,
//                                    "cannot specify 'out' as both a "
//                                    "positional and keyword argument");
//                            goto fail;
//                        }
//
//                        if (PyArray_Check(value)) {
//                            const char *name = "output array";
//                            PyArrayObject *value_arr = (PyArrayObject *)value;
//                            if (PyArray_FailUnlessWriteable(value_arr, name) < 0) {
//                                goto fail;
//                            }
//                            Py_INCREF(value);
//                            out_op[nin] = (PyArrayObject *)value;
//                        }
//                        else {
//                            PyErr_SetString(PyExc_TypeError,
//                                            "return arrays must be "
//                                            "of ArrayType");
//                            goto fail;
//                        }
//                        bad_arg = 0;
//                    }
//                    /* Allows the default output layout to be overridden */
//                    else if (strncmp(str,"order",5) == 0) {
//                        if (!PyArray_OrderConverter(value, out_order)) {
//                            goto fail;
//                        }
//                        bad_arg = 0;
//                    }
//                    break;
//                case 's':
//                    /* Allows a specific function inner loop to be selected */
//                    if (strncmp(str,"sig",3) == 0) {
//                        if (*out_typetup != NULL) {
//                            PyErr_SetString(PyExc_RuntimeError,
//                                    "cannot specify both 'sig' and 'dtype'");
//                            goto fail;
//                        }
//                        *out_typetup = value;
//                        Py_INCREF(value);
//                        bad_arg = 0;
//                    }
//                    else if (strncmp(str,"subok",5) == 0) {
//                        if (!PyBool_Check(value)) {
//                            PyErr_SetString(PyExc_TypeError,
//                                        "'subok' must be a boolean");
//                            goto fail;
//                        }
//                        *out_subok = (value == Py_True);
//                        bad_arg = 0;
//                    }
//                    break;
//                case 'w':
//                    /*
//                     * Provides a boolean array 'where=' mask if
//                     * out_wheremask is supplied.
//                     */
//                    if (out_wheremask != NULL &&
//                            strncmp(str,"where",5) == 0) {
//                        PyArray_Descr *dtype;
//                        dtype = PyArray_DescrFromType(NPY_BOOL);
//                        if (dtype == NULL) {
//                            goto fail;
//                        }
//                        *out_wheremask = (PyArrayObject *)PyArray_FromAny(
//                                                            value, dtype,
//                                                            0, 0, 0, NULL);
//                        if (*out_wheremask == NULL) {
//                            goto fail;
//                        }
//                        bad_arg = 0;
//                    }
//                    break;
//            }
//
//            if (bad_arg) {
//                char *format = "'%s' is an invalid keyword to ufunc '%s'";
//                PyErr_Format(PyExc_TypeError, format, str, ufunc_name);
//                goto fail;
//            }
//        }
//    }
//    Py_XDECREF(str_key_obj);
//
//    return 0;
//
//fail:
//    Py_XDECREF(str_key_obj);
//    Py_XDECREF(*out_extobj);
//    *out_extobj = NULL;
//    Py_XDECREF(*out_typetup);
//    *out_typetup = NULL;
//    if (out_wheremask != NULL) {
//        Py_XDECREF(*out_wheremask);
//        *out_wheremask = NULL;
//    }
//    return -1;
//}
//
////int _invoke_cuda()
//
//int
//PyUFunc_GeneralizedFunction(PyUFuncObject *ufunc,
//                        PyObject *args, PyObject *kwds,
//                        PyArrayObject **op)
//{
//    int nin, nout;
//    int i, idim, nop;
//    char *ufunc_name;
//    int retval = -1, subok = 1;
//    int needs_api = 0;
//
//    PyArray_Descr *dtypes[NPY_MAXARGS];
//
//    /* Use remapped axes for generalized ufunc */
//    int broadcast_ndim, op_ndim;
//    int op_axes_arrays[NPY_MAXARGS][NPY_MAXDIMS];
//    int *op_axes[NPY_MAXARGS];
//
//    npy_uint32 op_flags[NPY_MAXARGS];
//
//    NpyIter *iter = NULL;
//
//    /* These parameters come from extobj= or from a TLS global */
//    int buffersize = 0, errormask = 0;
//    PyObject *errobj = NULL;
//    int first_error = 1;
//
//    /* The selected inner loop */
//    PyUFuncGenericFunction innerloop = NULL;
//    void *innerloopdata = NULL;
//    /* The dimensions which get passed to the inner loop */
//    npy_intp inner_dimensions[NPY_MAXDIMS+1];
//    /* The strides which get passed to the inner loop */
//    npy_intp *inner_strides = NULL;
//
//    npy_intp *inner_strides_tmp, *ax_strides_tmp[NPY_MAXDIMS];
//    int core_dim_ixs_size, *core_dim_ixs;
//
//    /* The __array_prepare__ function to call for each output */
//    PyObject *arr_prep[NPY_MAXARGS];
//    /*
//     * This is either args, or args with the out= parameter from
//     * kwds added appropriately.
//     */
//    PyObject *arr_prep_args = NULL;
//
//    NPY_ORDER order = NPY_KEEPORDER;
//    /* Use the default assignment casting rule */
//    NPY_CASTING casting = NPY_DEFAULT_ASSIGN_CASTING;
//    /* When provided, extobj and typetup contain borrowed references */
//    PyObject *extobj = NULL, *type_tup = NULL;
//
//    if (ufunc == NULL) {
//        PyErr_SetString(PyExc_ValueError, "function not supported");
//        return -1;
//    }
//
//    nin = ufunc->nin;
//    nout = ufunc->nout;
//    nop = nin + nout;
//
//    ufunc_name = ufunc->name ? ufunc->name : "<unnamed ufunc>";
//
//    NPY_UF_DBG_PRINT1("\nEvaluating ufunc %s\n", ufunc_name);
//
//    /* Initialize all the operands and dtypes to NULL */
//    for (i = 0; i < nop; ++i) {
//        op[i] = NULL;
//        dtypes[i] = NULL;
//        arr_prep[i] = NULL;
//    }
//
//    NPY_UF_DBG_PRINT("Getting arguments\n");
//
//    /* Get all the arguments */
//    retval = get_ufunc_arguments(ufunc, args, kwds,
//                op, &order, &casting, &extobj,
//                &type_tup, &subok, NULL);
//    if (retval < 0) {
//        goto fail;
//    }
//
//    /* Figure out the number of dimensions needed by the iterator */
//    broadcast_ndim = 0;
//    for (i = 0; i < nin; ++i) {
//        int n = PyArray_NDIM(op[i]) - ufunc->core_num_dims[i];
//        if (n > broadcast_ndim) {
//            broadcast_ndim = n;
//        }
//    }
//    op_ndim = broadcast_ndim + ufunc->core_num_dim_ix;
//    if (op_ndim > NPY_MAXDIMS) {
//        PyErr_Format(PyExc_ValueError,
//                    "too many dimensions for generalized ufunc %s",
//                    ufunc_name);
//        retval = -1;
//        goto fail;
//    }
//
//    /* Fill in op_axes for all the operands */
//    core_dim_ixs_size = 0;
//    core_dim_ixs = ufunc->core_dim_ixs;
//    for (i = 0; i < nop; ++i) {
//        int n;
//        if (op[i]) {
//            /*
//             * Note that n may be negative if broadcasting
//             * extends into the core dimensions.
//             */
//            n = PyArray_NDIM(op[i]) - ufunc->core_num_dims[i];
//        }
//        else {
//            n = broadcast_ndim;
//        }
//        /* Broadcast all the unspecified dimensions normally */
//        for (idim = 0; idim < broadcast_ndim; ++idim) {
//            if (idim >= broadcast_ndim - n) {
//                op_axes_arrays[i][idim] = idim - (broadcast_ndim - n);
//            }
//            else {
//                op_axes_arrays[i][idim] = -1;
//            }
//        }
//        /* Use the signature information for the rest */
//        for (idim = broadcast_ndim; idim < op_ndim; ++idim) {
//            op_axes_arrays[i][idim] = -1;
//        }
//        for (idim = 0; idim < ufunc->core_num_dims[i]; ++idim) {
//            if (n + idim >= 0) {
//                op_axes_arrays[i][broadcast_ndim + core_dim_ixs[idim]] =
//                                                                    n + idim;
//            }
//            else {
//                op_axes_arrays[i][broadcast_ndim + core_dim_ixs[idim]] = -1;
//            }
//        }
//        core_dim_ixs_size += ufunc->core_num_dims[i];
//        core_dim_ixs += ufunc->core_num_dims[i];
//        op_axes[i] = op_axes_arrays[i];
//    }
//
//    /* Get the buffersize, errormask, and error object globals */
//    if (extobj == NULL) {
//        if (PyUFunc_GetPyValues(ufunc_name,
//                                &buffersize, &errormask, &errobj) < 0) {
//            retval = -1;
//            goto fail;
//        }
//    }
//    else {
//        if (_extract_pyvals(extobj, ufunc_name,
//                                &buffersize, &errormask, &errobj) < 0) {
//            retval = -1;
//            goto fail;
//        }
//    }
//
//    NPY_UF_DBG_PRINT("Finding inner loop\n");
//
//
//    retval = ufunc->type_resolver(ufunc, casting,
//                            op, type_tup, dtypes);
//    if (retval < 0) {
//        goto fail;
//    }
//    /* For the generalized ufunc, we get the loop right away too */
//    retval = ufunc->legacy_inner_loop_selector(ufunc, dtypes,
//                                    &innerloop, &innerloopdata, &needs_api);
//    if (retval < 0) {
//        goto fail;
//    }
//
//#if NPY_UF_DBG_TRACING
//    printf("input types:\n");
//    for (i = 0; i < nin; ++i) {
//        PyObject_Print((PyObject *)dtypes[i], stdout, 0);
//        printf(" ");
//    }
//    printf("\noutput types:\n");
//    for (i = nin; i < nop; ++i) {
//        PyObject_Print((PyObject *)dtypes[i], stdout, 0);
//        printf(" ");
//    }
//    printf("\n");
//#endif
//
////    if (subok) {
////        /*
////         * Get the appropriate __array_prepare__ function to call
////         * for each output
////         */
////        _find_array_prepare(args, kwds, arr_prep, nin, nout);
////
////        /* Set up arr_prep_args if a prep function was needed */
////        for (i = 0; i < nout; ++i) {
////            if (arr_prep[i] != NULL && arr_prep[i] != Py_None) {
////                arr_prep_args = make_arr_prep_args(nin, args, kwds);
////                break;
////            }
////        }
////    }
//
//    /*
//     * Set up the iterator per-op flags.  For generalized ufuncs, we
//     * can't do buffering, so must COPY or UPDATEIFCOPY.
//     */
//    for (i = 0; i < nin; ++i) {
//        op_flags[i] = NPY_ITER_READONLY|
//                      NPY_ITER_COPY|
//                      NPY_ITER_ALIGNED;
//    }
//    for (i = nin; i < nop; ++i) {
//        op_flags[i] = NPY_ITER_READWRITE|
//                      NPY_ITER_UPDATEIFCOPY|
//                      NPY_ITER_ALIGNED|
//                      NPY_ITER_ALLOCATE|
//                      NPY_ITER_NO_BROADCAST;
//    }
//
//    /* Create the iterator */
//    iter = NpyIter_AdvancedNew(nop, op, NPY_ITER_MULTI_INDEX|
//                                      NPY_ITER_REFS_OK|
//                                      NPY_ITER_REDUCE_OK,
//                           order, NPY_UNSAFE_CASTING, op_flags,
//                           dtypes, op_ndim, op_axes, NULL, 0);
//    if (iter == NULL) {
//        retval = -1;
//        goto fail;
//    }
//
//    /* Fill in any allocated outputs */
//    for (i = nin; i < nop; ++i) {
//        if (op[i] == NULL) {
//            op[i] = NpyIter_GetOperandArray(iter)[i];
//            Py_INCREF(op[i]);
//        }
//    }
//
//    /*
//     * Set up the inner strides array. Because we're not doing
//     * buffering, the strides are fixed throughout the looping.
//     */
//    inner_strides = (npy_intp *)PyArray_malloc(
//                        NPY_SIZEOF_INTP * (nop+core_dim_ixs_size));
//    /* The strides after the first nop match core_dim_ixs */
//    core_dim_ixs = ufunc->core_dim_ixs;
//    inner_strides_tmp = inner_strides + nop;
//    for (idim = 0; idim < ufunc->core_num_dim_ix; ++idim) {
//        ax_strides_tmp[idim] = NpyIter_GetAxisStrideArray(iter,
//                                                broadcast_ndim+idim);
//        if (ax_strides_tmp[idim] == NULL) {
//            retval = -1;
//            goto fail;
//        }
//    }
//    for (i = 0; i < nop; ++i) {
//        for (idim = 0; idim < ufunc->core_num_dims[i]; ++idim) {
//            inner_strides_tmp[idim] = ax_strides_tmp[core_dim_ixs[idim]][i];
//        }
//
//        core_dim_ixs += ufunc->core_num_dims[i];
//        inner_strides_tmp += ufunc->core_num_dims[i];
//    }
//
//    /* Set up the inner dimensions array */
//    if (NpyIter_GetShape(iter, inner_dimensions) != NPY_SUCCEED) {
//        retval = -1;
//        goto fail;
//    }
//    /* Move the core dimensions to start at the second element */
//    memmove(&inner_dimensions[1], &inner_dimensions[broadcast_ndim],
//                        NPY_SIZEOF_INTP * ufunc->core_num_dim_ix);
//
//    /* Remove all the core dimensions from the iterator */
//    for (i = 0; i < ufunc->core_num_dim_ix; ++i) {
//        if (NpyIter_RemoveAxis(iter, broadcast_ndim) != NPY_SUCCEED) {
//            retval = -1;
//            goto fail;
//        }
//    }
//    if (NpyIter_RemoveMultiIndex(iter) != NPY_SUCCEED) {
//        retval = -1;
//        goto fail;
//    }
//    if (NpyIter_EnableExternalLoop(iter) != NPY_SUCCEED) {
//        retval = -1;
//        goto fail;
//    }
//
//    /*
//     * The first nop strides are for the inner loop (but only can
//     * copy them after removing the core axes
//     */
//    memcpy(inner_strides, NpyIter_GetInnerStrideArray(iter),
//                                    NPY_SIZEOF_INTP * nop);
//
//#if 0
//    printf("strides: ");
//    for (i = 0; i < nop+core_dim_ixs_size; ++i) {
//        printf("%d ", (int)inner_strides[i]);
//    }
//    printf("\n");
//#endif
//
//    /* Start with the floating-point exception flags cleared */
//    PyUFunc_clearfperr();
//
//    NPY_UF_DBG_PRINT("Executing inner loop\n");
//
////    _invoke_cuda(op, inner_dimensions, inner_strides, nop);
//
////    /* Do the ufunc loop */
////    if (NpyIter_GetIterSize(iter) != 0) {
////        NpyIter_IterNextFunc *iternext;
////        char **dataptr;
////        npy_intp *count_ptr;
////
////        /* Get the variables needed for the loop */
////        iternext = NpyIter_GetIterNext(iter, NULL);
////        if (iternext == NULL) {
////            NpyIter_Deallocate(iter);
////            retval = -1;
////            goto fail;
////        }
////        dataptr = NpyIter_GetDataPtrArray(iter);
////        count_ptr = NpyIter_GetInnerLoopSizePtr(iter);
////
////        do {
////            inner_dimensions[0] = *count_ptr;
////            innerloop(dataptr, inner_dimensions, inner_strides, innerloopdata);
////        } while (iternext(iter));
////    }
//
//    /* Check whether any errors occurred during the loop */
//    if (PyErr_Occurred() || (errormask &&
//            PyUFunc_checkfperr(errormask, errobj, &first_error))) {
//        retval = -1;
//        goto fail;
//    }
//
//    PyArray_free(inner_strides);
//    NpyIter_Deallocate(iter);
//    /* The caller takes ownership of all the references in op */
//    for (i = 0; i < nop; ++i) {
//        Py_XDECREF(dtypes[i]);
//        Py_XDECREF(arr_prep[i]);
//    }
//    Py_XDECREF(errobj);
//    Py_XDECREF(type_tup);
//    Py_XDECREF(arr_prep_args);
//
//    NPY_UF_DBG_PRINT("Returning Success\n");
//
//    return 0;
//
//fail:
//    NPY_UF_DBG_PRINT1("Returning failure code %d\n", retval);
//    if (inner_strides) {
//        PyArray_free(inner_strides);
//    }
//    if (iter != NULL) {
//        NpyIter_Deallocate(iter);
//    }
//    for (i = 0; i < nop; ++i) {
//        Py_XDECREF(op[i]);
//        op[i] = NULL;
//        Py_XDECREF(dtypes[i]);
//        Py_XDECREF(arr_prep[i]);
//    }
//    Py_XDECREF(errobj);
//    Py_XDECREF(type_tup);
//    Py_XDECREF(arr_prep_args);
//
//    return retval;
//}