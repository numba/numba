/*
 * Copyright (c) 2012 Continuum Analytics, Inc. 
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither the name of this software, nor the names of its 
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* Python include */
#include "Python.h"
#include "numpy/ndarrayobject.h"
#include "numpy/ufuncobject.h"

#if PY_MAJOR_VERSION >= 3
#define IS_PY3K
#endif

#define UNARY_LOOP\
    char *ip1 = args[0], *op1 = args[1];\
    npy_intp is1 = steps[0], os1 = steps[1];\
    npy_intp n = dimensions[0];\
    npy_intp i;\
    for(i = 0; i < n; i++, ip1 += is1, op1 += os1)

typedef npy_double cunaryfunc(npy_double);

NPY_NO_EXPORT void
MyUFunc_D_D(char **args, npy_intp *dimensions, npy_intp *steps, void *func)
{
    cunaryfunc *f = (cunaryfunc *)func;
    UNARY_LOOP {
        npy_double in1 = *(npy_double *)ip1;
        npy_double *out = (npy_double *)op1;
        *out = f(in1);
    }
}

typedef npy_double funaryfunc_werror(npy_double *, npy_int32 *);

NPY_NO_EXPORT void
MyForUFunc_D_D(char **args, npy_intp *dimensions, npy_intp *steps, void *func)
{
    funaryfunc_werror *f = (funaryfunc_werror *)func;
    UNARY_LOOP {
        npy_double *in1 = (npy_double *)ip1;
        npy_double *out = (npy_double *)op1;
        npy_int32 err;
        *out = f(in1, &err);
        if (err!=0) *out=NPY_NAN;
    }
}

PyUFuncGenericFunction funcs[1] = {MyUFunc_D_D};
static char types[2] = {NPY_DOUBLE, NPY_DOUBLE};

PyUFuncGenericFunction funcs2[1] = {MyForUFunc_D_D};

static PyObject *
ufunc_from_ptr(PyObject *self, PyObject *args)
{

    Py_ssize_t func_ptr;
    int type = 0;
    char *func_name = "temp"; 
    void **data; 
    PyObject *ret;

    /* FIXME:  This will not be freed */    
    data = (void **)malloc(sizeof(void *));

    if (!PyArg_ParseTuple(args, "n|is", &func_ptr, &type, &func_name)) return NULL;
    data[0] = (void *)func_ptr;
    if (type == 0) {
        double(*func)(double);
        func = data[0];
        printf("%f %f ** \n" , func(4.3), func(0.0));    
        ret = PyUFunc_FromFuncAndData(funcs, data, types, 1, 1, 1, PyUFunc_None, func_name, "doc", 0);
    }
    else { /* dgamln like function from special/amos */
        double val = 4.3;
        npy_int32 err;
        double(*func2)(double *, npy_int32 *);
        func2 = data[0];
        printf("Hello there...%p\n", func2);
        printf("%f** \n", func2(&val, &err));
        printf("%d\n", (int)err);               
        
        ret = PyUFunc_FromFuncAndData(funcs2, data, types, 1, 1, 1, PyUFunc_None, func_name, "doc", 0);
    }

    return ret;
}

static PyObject *
get_libc_file_addrs(PyObject *self, PyObject *args)
{
    PyObject *result = NULL, *in = NULL, *out = NULL, *err = NULL;
    in = PyLong_FromVoidPtr(&stdin);
    out = PyLong_FromVoidPtr(&stdout);
    err = PyLong_FromVoidPtr(&stderr);
    if (!(in && out && err))
        goto error;

    result = PyTuple_Pack(3, in, out, err);

error:
    Py_XDECREF(in);
    Py_XDECREF(out);
    Py_XDECREF(err);
    return result;
}

static PyObject *
sizeof_py_ssize_t(PyObject *self, PyObject *args)
{
    return PyInt_FromSize_t(sizeof(Py_ssize_t));
}

static PyMethodDef ext_methods[] = {

#ifdef IS_PY3K
    {"make_ufunc", (PyCFunction) ufunc_from_ptr, METH_VARARGS, NULL},
    {"sizeof_py_ssize_t", (PyCFunction) sizeof_py_ssize_t, METH_NOARGS, NULL},
    {"get_libc_file_addrs", (PyCFunction) get_libc_file_addrs, METH_NOARGS, NULL},

#else
    {"make_ufunc", ufunc_from_ptr, METH_VARARGS},
    {"sizeof_py_ssize_t", sizeof_py_ssize_t, METH_NOARGS},
    {"get_libc_file_addrs", get_libc_file_addrs, METH_NOARGS},
#endif
    { NULL }
};


#if IS_PY3K

struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "_ext",
    NULL,
    -1,
    ext_methods,
    NULL, NULL, NULL, NULL
};

PyObject *
PyInit__ext(void)
{
    import_array();
    import_umath();

    PyObject *module = PyModule_Create( &module_def );
    return module;
}

#else

PyMODINIT_FUNC
init_ext(void)
{
    import_array();
    import_umath();

    PyObject *module = Py_InitModule("_ext", ext_methods);
    return;
}

#endif
