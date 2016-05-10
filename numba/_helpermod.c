/*
Expose all functions as pointers in a dedicated C extension.
*/

#define NUMBA_EXPORT_FUNC(_rettype) static _rettype
#define NUMBA_EXPORT_DATA(_vartype) static _vartype

/* Import _pymodule.h first, for a recent _POSIX_C_SOURCE */
#include "_pymodule.h"
#include <math.h>
#include "_helperlib.c"

static PyObject *
build_c_helpers_dict(void)
{
    PyObject *dct = PyDict_New();
    if (dct == NULL)
        goto error;

#define _declpointer(name, value) do {                 \
    PyObject *o = PyLong_FromVoidPtr(value);           \
    if (o == NULL) goto error;                         \
    if (PyDict_SetItemString(dct, name, o)) {          \
        Py_DECREF(o);                                  \
        goto error;                                    \
    }                                                  \
    Py_DECREF(o);                                      \
} while (0)

#define declmethod(func) _declpointer(#func, &numba_##func)

#define declpointer(ptr) _declpointer(#ptr, &numba_##ptr)

    declmethod(sdiv);
    declmethod(srem);
    declmethod(udiv);
    declmethod(urem);
    declmethod(frexp);
    declmethod(frexpf);
    declmethod(ldexp);
    declmethod(ldexpf);
    declmethod(cpow);
    declmethod(erf);
    declmethod(erff);
    declmethod(erfc);
    declmethod(erfcf);
    declmethod(gamma);
    declmethod(gammaf);
    declmethod(lgamma);
    declmethod(lgammaf);
    declmethod(complex_adaptor);
    declmethod(adapt_ndarray);
    declmethod(ndarray_new);
    declmethod(extract_record_data);
    declmethod(get_buffer);
    declmethod(adapt_buffer);
    declmethod(release_buffer);
    declmethod(extract_np_datetime);
    declmethod(create_np_datetime);
    declmethod(extract_np_timedelta);
    declmethod(create_np_timedelta);
    declmethod(recreate_record);
    declmethod(fptoui);
    declmethod(fptouif);
    declmethod(gil_ensure);
    declmethod(gil_release);
    declmethod(fatal_error);
    declmethod(py_type);
    declmethod(unpack_slice);
    declmethod(do_raise);
    declmethod(unpickle);
    declmethod(rnd_shuffle);
    declmethod(rnd_init);
    declmethod(poisson_ptrs);
    declmethod(attempt_nocopy_reshape);
    declmethod(get_list_private_data);
    declmethod(set_list_private_data);
    declmethod(reset_list_private_data);
    declmethod(get_pyobject_private_data);
    declmethod(set_pyobject_private_data);
    declmethod(reset_pyobject_private_data);
    declmethod(xxgemm);
    declmethod(xxgemv);
    declmethod(xxdot);
    declmethod(xxgetrf);
    declmethod(xxgetri);
    declmethod(xxpotrf);
    declmethod(ez_rgeev);
    declmethod(ez_cgeev);
    declmethod(ez_gesdd);
    declmethod(ez_geqrf);
    declmethod(ez_xxgqr);

    declpointer(py_random_state);
    declpointer(np_random_state);

#define MATH_UNARY(F, R, A) declmethod(F);
#define MATH_BINARY(F, R, A, B) declmethod(F);
    #include "mathnames.h"
#undef MATH_UNARY
#undef MATH_BINARY

#undef declmethod
    return dct;
error:
    Py_XDECREF(dct);
    return NULL;
}

static PyMethodDef ext_methods[] = {
    { "rnd_get_state", (PyCFunction) _numba_rnd_get_state, METH_O, NULL },
    { "rnd_seed", (PyCFunction) _numba_rnd_seed, METH_VARARGS, NULL },
    { "rnd_set_state", (PyCFunction) _numba_rnd_set_state, METH_VARARGS, NULL },
    { "rnd_shuffle", (PyCFunction) _numba_rnd_shuffle, METH_O, NULL },
    { NULL },
};

/*
 * These functions are exported by the module's DLL, to exercise ctypes / cffi
 * without relying on libc availability (see https://bugs.python.org/issue23606)
 */

PyAPI_FUNC(double) _numba_test_sin(double x);
PyAPI_FUNC(double) _numba_test_cos(double x);
PyAPI_FUNC(double) _numba_test_exp(double x);
PyAPI_FUNC(void) _numba_test_vsquare(int n, double *x, double *out);

double _numba_test_sin(double x)
{
    return sin(x);
}

double _numba_test_cos(double x)
{
    return cos(x);
}

double _numba_test_exp(double x)
{
    return exp(x);
}

void _numba_test_vsquare(int n, double *x, double *out)
{
    int i;
    for (i = 0; i < n; i++)
        out[i] = pow(x[i], 2.0);
}

void _numba_test_vcube(int n, double *x, double *out)
{
    int i;
    for (i = 0; i < n; i++)
        out[i] = pow(x[i], 3.0);
}


MOD_INIT(_helperlib) {
    PyObject *m;
    MOD_DEF(m, "_helperlib", "No docs", ext_methods)
    if (m == NULL)
        return MOD_ERROR_VAL;

    import_array();

    PyModule_AddObject(m, "c_helpers", build_c_helpers_dict());
    PyModule_AddIntConstant(m, "long_min", LONG_MIN);
    PyModule_AddIntConstant(m, "long_max", LONG_MAX);
    PyModule_AddIntConstant(m, "py_buffer_size", sizeof(Py_buffer));
    PyModule_AddIntConstant(m, "py_gil_state_size", sizeof(PyGILState_STATE));

    if (_numba_rnd_random_seed(&numba_py_random_state) ||
        _numba_rnd_random_seed(&numba_np_random_state))
        return MOD_ERROR_VAL;

    return MOD_SUCCESS_VAL(m);
}
