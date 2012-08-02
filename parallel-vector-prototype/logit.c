#include "Python.h"
#include "math.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"
#include "parallel_dispatch.h"

/*
 * single_type_logit.c
 * This is the C code for creating your own
 * Numpy ufunc for a logit function.
 *
 * In this code we only define the ufunc for
 * a single dtype. The computations that must
 * be replaced to create a ufunc for
 * a different funciton are marked with BEGIN
 * and END.
 *
 * Details explaining the Python-C API can be found under
 * 'Extending and Embedding' and 'Python/C API' at
 * docs.python.org .
 */

static PyMethodDef LogitMethods[] = {
        {NULL, NULL, 0, NULL}
};

/* The loop definition must precede the PyMODINIT_FUNC. */
/* Original
static void double_logit(char **args, npy_intp *dimensions,
                            npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in = args[0], *out = args[1];
    npy_intp in_step = steps[0], out_step = steps[1];

    double tmp;

    for (i = 0; i < n; i++) {
        // BEGIN main ufunc computation
        tmp = *(double *)in;
        tmp /= 1-tmp;
        *((double *)out) = log(tmp);
        // END main ufunc computation

        in += in_step;
        out += out_step;
    }
}
*/

static double logit(double x)
{
    return log(x/(1-x)) + sin(x) + 2 * cos(x);
}

static char primetest(int x)
{
    int i;
    if( x == 2 )
        return 1;
    if( x % 2 == 0 )
        return 0; // even --> not prime
    //for (i = 3; i < ceil(sqrt(x)); i+=2) { // I want it SLOWER
    for (i = 3; i < x; i+=2) {
        if ( x % i == 0 ) return 0;
    }
    return 1;
}

static void double_logit(char **args, npy_intp *dimensions,
                            npy_intp* steps, void* data)
{
    parallel_ufunc(logit, ufunc_worker_d_d, args, dimensions, steps, data);
}

static void int_primetest(char **args, npy_intp *dimensions,
                            npy_intp* steps, void* data)
{
    parallel_ufunc(primetest, ufunc_worker_i_b, args, dimensions, steps, data);
}

/*This a pointer to the above function*/
PyUFuncGenericFunction logit_funcs[1] = {&double_logit};
PyUFuncGenericFunction primetest_funcs[1] = {&int_primetest};

/* These are the input and return dtypes of logit.*/
static char logit_types[2] = {NPY_DOUBLE, NPY_DOUBLE};
static char primetest_types[2] = {NPY_INT, NPY_BOOL};

static void *data[1] = {NULL};

#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "npufunc",
    NULL,
    -1,
    LogitMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyObject *PyInit_npufunc(void)
{
    PyObject *m, *logit, *primetest, *d;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }

    import_array();
    import_umath();

    logit = PyUFunc_FromFuncAndData(logit_funcs, data, logit_types, 1, 1, 1,
                                    PyUFunc_None, "logit",
                                    "logit_docstring", 0);

    primetest = PyUFunc_FromFuncAndData(primetest_funcs, data, primetest_types,
                                        1, 1, 1,
                                        PyUFunc_None, "primetest",
                                        "primetest_docstring", 0);

    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "logit", logit);
    PyDict_SetItemString(d, "primetest", primetest);

    Py_DECREF(logit);
    Py_DECREF(primetest);

    return m;
}
#else
PyMODINIT_FUNC initnpufunc(void)
{
    PyObject *m, *logit, *primetest, *d;


    m = Py_InitModule("npufunc", LogitMethods);
    if (m == NULL) {
        return;
    }

    import_array();
    import_umath();

    logit = PyUFunc_FromFuncAndData(logit_funcs, data, logit_types, 1, 1, 1,
                                    PyUFunc_None, "logit",
                                    "logit_docstring", 0);
    primetest = PyUFunc_FromFuncAndData(primetest_funcs, data, primetest_types,
                                        1, 1, 1,
                                        PyUFunc_None, "primetest",
                                        "primetest_docstring", 0);

    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "logit", logit);
    PyDict_SetItemString(d, "primetest", primetest);
    Py_DECREF(logit);
    Py_DECREF(primetest);
}
#endif
