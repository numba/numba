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

typedef npy_cdouble cunaryfunc(npy_cdouble);

NPY_NO_EXPORT void
MyUFunc_D_D(char **args, npy_intp *dimensions, npy_intp *steps, void *func)
{
    cunaryfunc *f = (cunaryfunc *)func;
    UNARY_LOOP {
        npy_cdouble in1 = *(npy_cdouble *)ip1;
        npy_cdouble *out = (npy_cdouble *)op1;
        *out = f(in1);
    }
}

PyUFuncGenericFunction funcs[1] = {MyUFunc_D_D};
static char types[2] = {NPY_DOUBLE, NPY_DOUBLE};

static PyObject *
ufunc_from_ptr(PyObject *self, PyObject *args)
{

    Py_ssize_t func_ptr; 
    char *func_name = "temp"; 
    void **data; 
    PyObject *ret;
    double(*func)(double);

    /* FIXME:  This will not be freed */    
    data = (void **)malloc(sizeof(void **));

    if (!PyArg_ParseTuple(args, "n|s", &func_ptr, &func_name)) return NULL;
    data[0] = (void *)func_ptr;
    func = data[0];
    printf("%f" , func(4.3));
    ret = PyUFunc_FromFuncAndData(funcs, data, types, 1, 1, 1, PyUFunc_None, func_name, "doc", 0);

    return ret;
}


static PyMethodDef ext_methods[] = {

#ifdef IS_PY3K
    {"make_ufunc", (PyCFunction) ufunc_from_ptr, METH_VARARGS, NULL},
#else
    {"make_ufunc", ufunc_from_ptr, METH_VARARGS},
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
