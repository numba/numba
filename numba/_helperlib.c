#include "_pymodule.h"
#include <stdio.h>
#include <math.h>
#include "_math_c99.h"
#ifdef _MSC_VER
    #define int64_t signed __int64
    #define uint64_t unsigned __int64
#else
    #include <stdint.h>
#endif


/* provide 64-bit division function to 32-bit platforms */
static
int64_t Numba_sdiv(int64_t a, int64_t b) {
    return a / b;
}

static
uint64_t Numba_udiv(uint64_t a, uint64_t b) {
    return a / b;
}

/* provide 64-bit remainder function to 32-bit platforms */
static
int64_t Numba_srem(int64_t a, int64_t b) {
    return a % b;
}

static
uint64_t Numba_urem(uint64_t a, uint64_t b) {
    return a % b;
}

/* provide complex power */
static
void Numba_cpow(Py_complex *a, Py_complex *b, Py_complex *c) {
    *c = _Py_c_pow(*a, *b);
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



#define EXPOSE(Fn, Sym) static void* Sym(){return PyLong_FromVoidPtr(&Fn);}
EXPOSE(Numba_sdiv, get_sdiv)
EXPOSE(Numba_srem, get_srem)
EXPOSE(Numba_udiv, get_udiv)
EXPOSE(Numba_urem, get_urem)
EXPOSE(Numba_cpow, get_cpow)
EXPOSE(Numba_to_complex, get_complex_adaptor)
#undef EXPOSE

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
    declmethod(get_sdiv),
    declmethod(get_srem),
    declmethod(get_udiv),
    declmethod(get_urem),
    declmethod(get_cpow),
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
