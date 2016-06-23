/*
 * This file contains exports of Numpy math functions needed by numba.
 */

#include "_pymodule.h"
#include <numpy/npy_math.h>
#include <math.h>


/*
 * Map Numpy C function symbols to their addresses.
 */

struct npymath_entry {
    const char *name;
    void *func;
};

#define NPYMATH_SYMBOL(name) \
    { "npy_" #name, (void*) npy_##name }

static struct npymath_entry npymath_exports[] = {
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

    NPYMATH_SYMBOL(modf),

    /* float functions */
    NPYMATH_SYMBOL(sinf),
    NPYMATH_SYMBOL(cosf),
    NPYMATH_SYMBOL(tanf),
    NPYMATH_SYMBOL(asinf),
    NPYMATH_SYMBOL(acosf),
    NPYMATH_SYMBOL(atanf),

    NPYMATH_SYMBOL(sinhf),
    NPYMATH_SYMBOL(coshf),
    NPYMATH_SYMBOL(tanhf),
    NPYMATH_SYMBOL(asinhf),
    NPYMATH_SYMBOL(acoshf),
    NPYMATH_SYMBOL(atanhf),
    NPYMATH_SYMBOL(hypotf),

    NPYMATH_SYMBOL(expf),
    NPYMATH_SYMBOL(exp2f),
    NPYMATH_SYMBOL(expm1f),

    NPYMATH_SYMBOL(logf),
    NPYMATH_SYMBOL(log2f),
    NPYMATH_SYMBOL(log10f),
    NPYMATH_SYMBOL(log1pf),

    NPYMATH_SYMBOL(powf),
    NPYMATH_SYMBOL(sqrtf),

    NPYMATH_SYMBOL(atan2f),

    NPYMATH_SYMBOL(logaddexpf),
    NPYMATH_SYMBOL(logaddexp2f),
    NPYMATH_SYMBOL(nextafterf),
    NPYMATH_SYMBOL(spacingf),

    NPYMATH_SYMBOL(modff),
};

#undef NPYMATH_SYMBOL
