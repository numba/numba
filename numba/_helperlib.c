#include "_pymodule.h"
#include <stdio.h>
#include <math.h>

static
void Numba_cpow(Py_complex *a, Py_complex *b, Py_complex *c) {
    *c = _Py_c_pow(*a, *b);
}


static
void* get_cpow_pointer() {
    return PyLong_FromVoidPtr(&Numba_cpow);
}

static
int Numba_to_complex(PyObject* obj, Py_complex *out) {
    PyObject* fobj;
    if (PyComplex_Check(obj)) {
        out->real = PyComplex_RealAsDouble(obj);
        out->imag = PyComplex_ImagAsDouble(obj);
    } else {
        fobj = PyNumber_Float(obj);
        if (!fobj) return 0;
        out->real = PyFloat_AsDouble(fobj);
        out->imag = 0.;
        Py_DECREF(fobj);
    }
    return 1;
}

static
void* get_complex_adaptor() {
    return PyLong_FromVoidPtr(&Numba_to_complex);
}

/*
Define bridge for all math functions
*/
#define MATH_UNARY(F, R, A) static R Numba_##F(A a) { return F(a); }
#define MATH_BINARY(F, R, A, B) static R Numba_##F(A a, B b) \
                                       { return F(a, b); }
    #include "mathnames.inc"
#undef MATH_UNARY
#undef MATH_BINARY

/*
Expose all math functions
*/
#define MATH_UNARY(F, R, A) static void* get_##F() \
                            { return PyLong_FromVoidPtr(&Numba_##F);}
#define MATH_BINARY(F, R, A, B) MATH_UNARY(F, R, A)
    #include "mathnames.inc"
#undef MATH_UNARY
#undef MATH_BINARY

static PyMethodDef ext_methods[] = {
#define declmethod(func) { #func , ( PyCFunction )func , METH_VARARGS , NULL }
    declmethod(get_cpow_pointer),
    declmethod(get_complex_adaptor),

    /* Declare math exposer */
    #define MATH_UNARY(F, R, A) declmethod(get_##F),
    #define MATH_BINARY(F, R, A, B) MATH_UNARY(F, R, A)
        #include "mathnames.inc"
    #undef MATH_UNARY
    #undef MATH_BINARY
    { NULL },
#undef declmethod
};


MOD_INIT(_helperlib) {
    PyObject *m;
    MOD_DEF(m, "_helperlib", "No docs", ext_methods)
    if (m == NULL)
        return MOD_ERROR_VAL;

    return MOD_SUCCESS_VAL(m);
}
