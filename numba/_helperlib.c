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

/* Minimum PyBufferObject structure to hack inside it */
typedef struct {
    PyObject_HEAD
    PyObject *b_base;
    void *b_ptr;
    Py_ssize_t b_size;
    Py_ssize_t b_offset;
}  PyBufferObject_Hack;

/*
Get data address of record data buffer
*/
static
void* Numba_extract_record_data(PyObject *recordobj) {
    PyObject *attrdata;
    void *ptr;
    Py_buffer buf;

    attrdata = PyObject_GetAttrString(recordobj, "data");
    if (!attrdata) return NULL;

    if (-1 == PyObject_GetBuffer(attrdata, &buf, 0)){
        #if PY_MAJOR_VERSION >= 3
            return NULL;
        #else
            /* HACK!!! */
            /* In Python 2.6, it will report no buffer interface for record
               even though it should */
            PyBufferObject_Hack *hack;
            hack = (PyBufferObject_Hack*) attrdata;

            if (hack->b_base == NULL) {
                ptr = hack->b_ptr;
            } else {
                PyBufferProcs *bp;
                readbufferproc proc = NULL;

                bp = hack->b_base->ob_type->tp_as_buffer;
                /* FIXME Ignoring any flag.  Just give me the pointer */
                proc = (readbufferproc)bp->bf_getreadbuffer;
                if ((*proc)(hack->b_base, 0, &ptr) <= 0) {
                    return NULL;
                }
                ptr = (char*)ptr + hack->b_offset;
            }
        #endif
    } else {
        ptr = buf.buf;
    }
    Py_DECREF(attrdata);
    return ptr;
}

static
PyObject* Numba_recreate_record(void *pdata, int size, PyObject *dtype){
    /*
    import numpy
    buffer = memoryview(pdata, size)
    aryobj = numpy.array(buffer, dtype=(numpy.record, dtype))
    return aryobj[0]
    */
    PyObject *buffer = NULL;
    PyObject *numpy = NULL;
    PyObject *numpy_array = NULL;
    PyObject *numpy_record = NULL;
    PyObject *aryobj = NULL;
    PyObject *args = NULL;
    PyObject *kwargs = NULL;
    PyObject *dtypearg = NULL;
    PyObject *record = NULL;
    PyObject *index = NULL;

#if PY_MAJOR_VERSION >= 3
    buffer = PyMemoryView_FromMemory(pdata, size, PyBUF_WRITE);
#else
    buffer = PyBuffer_FromMemory(pdata, size);
#endif
    if (!buffer) goto CLEANUP;

    numpy = PyImport_ImportModuleNoBlock("numpy");
    if (!numpy) goto CLEANUP;

    numpy_array = PyObject_GetAttrString(numpy, "array");
    if (!numpy_array) goto CLEANUP;

    numpy_record = PyObject_GetAttrString(numpy, "record");
    if (!numpy_record) goto CLEANUP;

    args = Py_BuildValue("([O])", buffer);
    if (!args) goto CLEANUP;

    dtypearg = Py_BuildValue("(OO)", numpy_record, dtype);
    if (!dtypearg) goto CLEANUP;

    kwargs = Py_BuildValue("{sO}", "dtype", dtypearg);
    if (!kwargs) goto CLEANUP;

    aryobj = PyObject_Call(numpy_array, args, kwargs);
    if (!aryobj) goto CLEANUP;

    index = Py_BuildValue("i", 0);
    if (!index) goto CLEANUP;

    record = PyObject_GetItem(aryobj, index);

CLEANUP:
    Py_XDECREF(numpy);
    Py_XDECREF(numpy_array);
    Py_XDECREF(numpy_record);
    Py_XDECREF(aryobj);
    Py_XDECREF(args);
    Py_XDECREF(kwargs);
    Py_XDECREF(dtypearg);
    Py_XDECREF(buffer);
    Py_XDECREF(index);

    return record;
}

static
double Numba_round_even(double y) {
    double z = round(y);
    if (fabs(y-z) == 0.5) {
        /* halfway between two integers; use round-half-even */
        z = 2.0*round(y / 2.0);
    }
    return z;
}

static
float Numba_roundf_even(float y) {
    float z = roundf(y);
    if (fabsf(y-z) == 0.5) {
        /* halfway between two integers; use round-half-even */
        z = 2.0 * roundf(y / 2.0);
    }
    return z;
}

static
uint64_t Numba_fptoui(double x) {
    return (uint64_t)x;
}


#define EXPOSE(Fn, Sym) static void* Sym(){return PyLong_FromVoidPtr(&Fn);}
EXPOSE(Numba_sdiv, get_sdiv)
EXPOSE(Numba_srem, get_srem)
EXPOSE(Numba_udiv, get_udiv)
EXPOSE(Numba_urem, get_urem)
EXPOSE(Numba_cpow, get_cpow)
EXPOSE(Numba_to_complex, get_complex_adaptor)
EXPOSE(Numba_extract_record_data, get_extract_record_data)
EXPOSE(Numba_recreate_record, get_recreate_record)
EXPOSE(Numba_round_even, get_round_even)
EXPOSE(Numba_roundf_even, get_roundf_even)
EXPOSE(Numba_fptoui, get_fptoui)
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
#define MATH_BINARY(F, R, A, B) static void* get_##F() \
                            { return PyLong_FromVoidPtr(&Numba_##F);}
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
    declmethod(get_extract_record_data),
    declmethod(get_recreate_record),
    declmethod(get_round_even),
    declmethod(get_roundf_even),
    declmethod(get_fptoui),

    /* Declare math exposer */
    #define MATH_UNARY(F, R, A) declmethod(get_##F),
    #define MATH_BINARY(F, R, A, B) declmethod(get_##F),
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
