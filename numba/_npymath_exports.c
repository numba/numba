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
    NPYMATH_SYMBOL(exp2),
    NPYMATH_SYMBOL(log2),

    NPYMATH_SYMBOL(logaddexp),
    NPYMATH_SYMBOL(logaddexp2),
    NPYMATH_SYMBOL(nextafter),
    NPYMATH_SYMBOL(spacing),

    NPYMATH_SYMBOL(modf),

    /* float functions */
    NPYMATH_SYMBOL(exp2f),
    NPYMATH_SYMBOL(log2f),

    NPYMATH_SYMBOL(logaddexpf),
    NPYMATH_SYMBOL(logaddexp2f),
    NPYMATH_SYMBOL(nextafterf),
    NPYMATH_SYMBOL(spacingf),

    NPYMATH_SYMBOL(modff),
};

#undef NPYMATH_SYMBOL
