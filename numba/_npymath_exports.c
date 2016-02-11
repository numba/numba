/*
 * This is a sample module exposing numpy math functions needed by numba.
 *
 * The module unique content will be a property containing a vector of tuples.
 * Each tuple will hold (symbol_name, function_pointer).
 */


#include "_pymodule.h"
#include <numpy/npy_math.h>
#include <math.h>


/* Missing math function on windows prior to VS2015 */
#if defined(_WIN32) && _MSC_VER < 1900
    /* undef windows macros for the following */
    #undef ldexpf
    #undef frexpf

    static
    float ldexpf(float x, int exp) {
        return (float)ldexp(x, exp);
    }

    static
    float frexpf(float x, int *exp) {
        return (float)frexp(x, exp);
    }
#endif /* WIN32 and _MSC_VER < 1900 */

/* signbit is actually a macro, two versions will be exported as to let the
   macro do whatever magic it does for floats and for doubles */

npy_bool
ufunc_signbitf(npy_float a)
{
    return npy_signbit(a) != 0;
}

npy_bool
ufunc_signbit(npy_double a)
{
    return npy_signbit(a) != 0;
}

/* Some functions require being adapted from the ones in npymath for
   use in numpy loops. It is easier to do this at this point than having
   to write code generation for the equivalent code.

   The code here usually reflect what can be found in NumPy's
   funcs.inc.src.
*/

void
ufunc_cpowf(npy_cfloat *dst, npy_cfloat *a, npy_cfloat *b)
{
    float br = npy_crealf(*b);
    float bi = npy_cimagf(*b);
    float ar = npy_crealf(*a);
    float ai = npy_cimagf(*b);

    if (br == 0.0f && bi == 0.0f) {
        /* on a 0 exponent, result is 1.0 + 0.0i */
        *dst = npy_cpackf(1.0f, 0.0f);
        return;
    }

    if (ar == 0.0f && ai == 0.0f) {
        if (br > 0 && bi == 0) {
            *dst = npy_cpackf(0.0f, 0.0f);
        }
        else {
            /* NB: there are four complex zeros; c0 = (+-0, +-0), so that unlike
             *     for reals, c0**p, with `p` negative is in general
             *     ill-defined.
             *
             *     c0**z with z complex is also ill-defined.
             */
            *dst = npy_cpackf(NPY_NAN, NPY_NAN);

            /* Raise invalid */
            npy_set_floatstatus_invalid();
        }
        return;
    }

    /* note: in NumPy there are optimizations for integer
    *exponents. These are not present here...
    */

    *dst = npy_cpowf(*a, *b);
    return;
}


void
ufunc_cpow(npy_cdouble *dst, npy_cdouble *a, npy_cdouble *b)
{
    double br = npy_creal(*b);
    double bi = npy_cimag(*b);
    double ar = npy_creal(*a);
    double ai = npy_cimag(*b);

    if (br == 0.0 && bi == 0.0) {
        /* on a 0 exponent, result is 1.0 + 0.0i */
        *dst = npy_cpack(1.0, 0.0);
        return;
    }

    if (ar == 0.0 && ai == 0.0) {
        if (br > 0 && bi == 0) {
            *dst = npy_cpack(0.0, 0.0);
        }
        else {
            /* NB: there are four complex zeros; c0 = (+-0, +-0), so that unlike
             *     for reals, c0**p, with `p` negative is in general
             *     ill-defined.
             *
             *     c0**z with z complex is also ill-defined.
             */
            *dst = npy_cpack(NPY_NAN, NPY_NAN);

            /* Raise invalid */
            npy_set_floatstatus_invalid();
        }
        return;
    }

    /* note: in NumPy there are optimizations for integer
    *exponents. These are not present here...
    */

    *dst = npy_cpow(*a, *b);
    return;
}


/* Use this macros to wrap functions that on C have complex arguments.
   In numba complex numbers are passed by reference, and the return
   value is passed as a first arg. This is different in npy_math */

#define NUMBA_UNARY_FUNC_WRAP(func, type)                               \
    void npy_ ## func ## _wrapped(type* dst, type* op1)                 \
    {                                                                   \
        *dst = npy_ ## func(*op1);                                      \
    }

#define NUMBA_BINARY_FUNC_WRAP(func, type)                          \
    void npy_ ## func ## _wrapped(type* dst, type* op1, type* op2)  \
    {                                                               \
        *dst = npy_ ## func(*op1, *op2);                            \
    }


NUMBA_UNARY_FUNC_WRAP(cexpf, npy_cfloat);
NUMBA_UNARY_FUNC_WRAP(cexp, npy_cdouble);
NUMBA_UNARY_FUNC_WRAP(clogf, npy_cfloat);
NUMBA_UNARY_FUNC_WRAP(clog, npy_cdouble);
NUMBA_UNARY_FUNC_WRAP(csqrtf, npy_cfloat);
NUMBA_UNARY_FUNC_WRAP(csqrt, npy_cdouble);

NUMBA_UNARY_FUNC_WRAP(csinf, npy_cfloat);
NUMBA_UNARY_FUNC_WRAP(csin, npy_cdouble);
NUMBA_UNARY_FUNC_WRAP(ccosf, npy_cfloat);
NUMBA_UNARY_FUNC_WRAP(ccos, npy_cdouble);


struct npy_math_entry {
    const char* name;
    void* func;
};


#define NPYMATH_SYMBOL_EXPLICIT(name,function) \
    { "numba.npymath." #name, (void*) function }

#define NPYMATH_SYMBOL(name) \
    { "numba.npymath." #name, (void*) npy_##name }

#define NPYMATH_SYMBOL_WRAPPED(name) \
    { "numba.npymath." #name, (void*) npy_##name##_wrapped }

struct npy_math_entry exports[] = {
    /* double functions */
    NPYMATH_SYMBOL(sin),
    NPYMATH_SYMBOL(cos),
    NPYMATH_SYMBOL(tan),
    NPYMATH_SYMBOL(asin),
    NPYMATH_SYMBOL(acos),
    NPYMATH_SYMBOL(atan),

    NPYMATH_SYMBOL(sinh),
    NPYMATH_SYMBOL(cosh),
    NPYMATH_SYMBOL(tanh),
    NPYMATH_SYMBOL(asinh),
    NPYMATH_SYMBOL(acosh),
    NPYMATH_SYMBOL(atanh),
    NPYMATH_SYMBOL(hypot),

    NPYMATH_SYMBOL(exp),
    NPYMATH_SYMBOL(exp2),
    NPYMATH_SYMBOL(expm1),

    NPYMATH_SYMBOL(log),
    NPYMATH_SYMBOL(log2),
    NPYMATH_SYMBOL(log10),
    NPYMATH_SYMBOL(log1p),

    NPYMATH_SYMBOL(pow),
    NPYMATH_SYMBOL(sqrt),

    NPYMATH_SYMBOL(atan2),

    NPYMATH_SYMBOL(logaddexp),
    NPYMATH_SYMBOL(logaddexp2),
    NPYMATH_SYMBOL(nextafter),
    NPYMATH_SYMBOL(spacing),
    /* npy_ldexp and npy_frexp appear in npy_math.h past NumPy 1.9, so link
       directly to the math.h versions. */
    NPYMATH_SYMBOL_EXPLICIT(ldexp, ldexp),
    NPYMATH_SYMBOL_EXPLICIT(frexp, frexp),
    NPYMATH_SYMBOL_EXPLICIT(signbit, ufunc_signbit),
    NPYMATH_SYMBOL(modf),

    /* float functions */
    NPYMATH_SYMBOL(powf),
    NPYMATH_SYMBOL(expf),
    NPYMATH_SYMBOL(exp2f),
    NPYMATH_SYMBOL(logf),
    NPYMATH_SYMBOL(log2f),
    NPYMATH_SYMBOL(log10f),
    NPYMATH_SYMBOL(expm1f),
    NPYMATH_SYMBOL(log1pf),
    NPYMATH_SYMBOL(sinf),
    NPYMATH_SYMBOL(cosf),
    NPYMATH_SYMBOL(tanf),
    NPYMATH_SYMBOL(atan2f),
    NPYMATH_SYMBOL(hypotf),
    NPYMATH_SYMBOL(sqrtf),
    NPYMATH_SYMBOL(sinhf),
    NPYMATH_SYMBOL(coshf),
    NPYMATH_SYMBOL(asinf),
    NPYMATH_SYMBOL(acosf),
    NPYMATH_SYMBOL(atanf),
    NPYMATH_SYMBOL(atan2f),
    NPYMATH_SYMBOL(hypotf),
    NPYMATH_SYMBOL(sinhf),
    NPYMATH_SYMBOL(coshf),
    NPYMATH_SYMBOL(tanhf),
    NPYMATH_SYMBOL(asinhf),
    NPYMATH_SYMBOL(acoshf),
    NPYMATH_SYMBOL(atanhf),
    NPYMATH_SYMBOL(logaddexpf),
    NPYMATH_SYMBOL(logaddexp2f),
    NPYMATH_SYMBOL(nextafterf),
    NPYMATH_SYMBOL(spacingf),
    /* npy_ldexpf and npy_frexpf appear in npy_math.h past NumPy 1.9, so link
       directly to the math.h versions. */
    NPYMATH_SYMBOL_EXPLICIT(ldexpf, ldexpf),
    NPYMATH_SYMBOL_EXPLICIT(frexpf, frexpf),
    NPYMATH_SYMBOL_EXPLICIT(signbitf, ufunc_signbitf),

    NPYMATH_SYMBOL(modff),

    /* complex functions */
    NPYMATH_SYMBOL_EXPLICIT(cpow, ufunc_cpow),
    NPYMATH_SYMBOL_WRAPPED(cexp),
    NPYMATH_SYMBOL_WRAPPED(clog),
    NPYMATH_SYMBOL_WRAPPED(csqrt),
    NPYMATH_SYMBOL_WRAPPED(csin),
    NPYMATH_SYMBOL_WRAPPED(ccos),

    /* complex float functions */
    NPYMATH_SYMBOL_EXPLICIT(cpowf, ufunc_cpowf),
    NPYMATH_SYMBOL_WRAPPED(cexpf),
    NPYMATH_SYMBOL_WRAPPED(clogf),
    NPYMATH_SYMBOL_WRAPPED(csqrtf),
    NPYMATH_SYMBOL_WRAPPED(csinf),
    NPYMATH_SYMBOL_WRAPPED(ccosf),
};
#undef NPY_MATH_SYMBOL

PyObject*
create_symbol_list(void)
{
    /*
     * note: reference stealing at its best
     * returns a PyList with a tuple for each symbol. The PyList has one reference.
     */
    size_t count = sizeof(exports) / sizeof(exports[0]);
    PyObject* pylist = PyList_New(count);
    size_t i;

    for (i = 0; i < count; ++i) {
        /* create the tuple */
        PyObject* ptr = PyLong_FromVoidPtr(exports[i].func);
        PyObject* tuple = Py_BuildValue("(s,O)", exports[i].name, ptr);
        PyList_SET_ITEM(pylist, i, tuple);
        Py_XDECREF(ptr);
    }

    return pylist;
}

MOD_INIT(_npymath_exports) {
    PyObject *module;
    MOD_DEF(module, "_npymath_exports", "No docs", NULL)
    if (!module) {
        return MOD_ERROR_VAL;
    }

    PyModule_AddObject(module, "symbols", create_symbol_list());

    return MOD_SUCCESS_VAL(module);
}
