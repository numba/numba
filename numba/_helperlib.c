/*
 * Helper functions used by Numba at runtime.
 * This C file is meant to be included after defining the
 * NUMBA_EXPORT_FUNC() and NUMBA_EXPORT_DATA() macros.
 */

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
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>
#include <numpy/arrayscalars.h>

#include "_arraystruct.h"

/* For Numpy 1.6 */
#ifndef NPY_ARRAY_BEHAVED
    #define NPY_ARRAY_BEHAVED NPY_BEHAVED
#endif

/*
 * PRNG support.
 */

/* Magic Mersenne Twister constants */
#define MT_N 624
#define MT_M 397
#define MT_MATRIX_A 0x9908b0dfU
#define MT_UPPER_MASK 0x80000000U
#define MT_LOWER_MASK 0x7fffffffU

/* unsigned int is sufficient on modern machines as we only need 32 bits */
typedef struct {
    int index;
    unsigned int mt[MT_N];
    int has_gauss;
    double gauss;
} rnd_state_t;

NUMBA_EXPORT_DATA(rnd_state_t) numba_py_random_state;
NUMBA_EXPORT_DATA(rnd_state_t) numba_np_random_state;

/* Some code portions below from CPython's _randommodule.c, some others
   from Numpy's and Jean-Sebastien Roy's randomkit.c. */

NUMBA_EXPORT_FUNC(void)
numba_rnd_shuffle(rnd_state_t *state)
{
    int i;
    unsigned int y;

    for (i = 0; i < MT_N - MT_M; i++) {
        y = (state->mt[i] & MT_UPPER_MASK) | (state->mt[i+1] & MT_LOWER_MASK);
        state->mt[i] = state->mt[i+MT_M] ^ (y >> 1) ^
                       (-(int) (y & 1) & MT_MATRIX_A);
    }
    for (; i < MT_N - 1; i++) {
        y = (state->mt[i] & MT_UPPER_MASK) | (state->mt[i+1] & MT_LOWER_MASK);
        state->mt[i] = state->mt[i+(MT_M-MT_N)] ^ (y >> 1) ^
                       (-(int) (y & 1) & MT_MATRIX_A);
    }
    y = (state->mt[MT_N - 1] & MT_UPPER_MASK) | (state->mt[0] & MT_LOWER_MASK);
    state->mt[MT_N - 1] = state->mt[MT_M - 1] ^ (y >> 1) ^
                          (-(int) (y & 1) & MT_MATRIX_A);
}

/* Initialize mt[] with an integer seed */
NUMBA_EXPORT_FUNC(void)
numba_rnd_init(rnd_state_t *state, unsigned int seed)
{
    unsigned int pos;
    seed &= 0xffffffffU;

    /* Knuth's PRNG as used in the Mersenne Twister reference implementation */
    for (pos = 0; pos < MT_N; pos++) {
        state->mt[pos] = seed;
        seed = (1812433253U * (seed ^ (seed >> 30)) + pos + 1) & 0xffffffffU;
    }
    state->index = MT_N;
    state->has_gauss = 0;
    state->gauss = 0.0;
}

/* Perturb mt[] with a key array */
static void
rnd_init_by_array(rnd_state_t *state, unsigned int init_key[], size_t key_length)
{
    size_t i, j, k;
    unsigned int *mt = state->mt;

    numba_rnd_init(state, 19650218U);
    i = 1; j = 0;
    k = (MT_N > key_length ? MT_N : key_length);
    for (; k; k--) {
        mt[i] = (mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 30)) * 1664525U))
                 + init_key[j] + (unsigned int) j; /* non linear */
        mt[i] &= 0xffffffffU;
        i++; j++;
        if (i >= MT_N) { mt[0] = mt[MT_N - 1]; i = 1; }
        if (j >= key_length) j = 0;
    }
    for (k = MT_N - 1; k; k--) {
        mt[i] = (mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 30)) * 1566083941U))
                 - (unsigned int) i; /* non linear */
        mt[i] &= 0xffffffffU;
        i++;
        if (i >= MT_N) { mt[0] = mt[MT_N - 1]; i=1; }
    }

    mt[0] = 0x80000000U; /* MSB is 1; ensuring non-zero initial array */
    state->index = MT_N;
    state->has_gauss = 0;
    state->gauss = 0.0;
}

/* Random-initialize the given state (for use at startup) */
NUMBA_EXPORT_FUNC(int)
_numba_rnd_random_seed(rnd_state_t *state)
{
    PyObject *timemod, *timeobj;
    double timeval;
    unsigned int seed, rshift;

    timemod = PyImport_ImportModuleNoBlock("time");
    if (!timemod)
        return -1;
    timeobj = PyObject_CallMethod(timemod, "time", NULL);
    Py_DECREF(timemod);
    timeval = PyFloat_AsDouble(timeobj);
    Py_DECREF(timeobj);
    if (timeval == -1 && PyErr_Occurred())
        return -1;
    /* Mix in seconds and nanoseconds */
    seed = (unsigned) timeval ^ (unsigned) (timeval * 1e9);
#ifndef _WIN32
    seed ^= getpid();
#endif
    /* Address space randomization bits: take MSBs of various pointers.
     * It is counter-productive to shift by 32, since the virtual address
     * space width is usually less than 64 bits (48 on x86-64).
     */
    rshift = sizeof(void *) > 4 ? 16 : 0;
    seed ^= (Py_uintptr_t) &timemod >> rshift;
    seed += (Py_uintptr_t) &PyObject_CallMethod >> rshift;
    numba_rnd_init(state, seed);
    return 0;
}

/* Python-exposed helpers for state management */
static int
rnd_state_converter(PyObject *obj, rnd_state_t **state)
{
    *state = (rnd_state_t *) PyLong_AsVoidPtr(obj);
    return (*state != NULL || !PyErr_Occurred());
}

NUMBA_EXPORT_FUNC(PyObject *)
_numba_rnd_shuffle(PyObject *self, PyObject *arg)
{
    rnd_state_t *state;
    if (!rnd_state_converter(arg, &state))
        return NULL;
    numba_rnd_shuffle(state);
    Py_RETURN_NONE;
}

NUMBA_EXPORT_FUNC(PyObject *)
_numba_rnd_set_state(PyObject *self, PyObject *args)
{
    int i, index;
    rnd_state_t *state;
    PyObject *tuplearg, *intlist;

    if (!PyArg_ParseTuple(args, "O&O!:rnd_set_state",
                          rnd_state_converter, &state,
                          &PyTuple_Type, &tuplearg))
        return NULL;
    if (!PyArg_ParseTuple(tuplearg, "iO!", &index, &PyList_Type, &intlist))
        return NULL;
    if (PyList_GET_SIZE(intlist) != MT_N) {
        PyErr_SetString(PyExc_ValueError, "list object has wrong size");
        return NULL;
    }
    state->index = index;
    for (i = 0; i < MT_N; i++) {
        PyObject *v = PyList_GET_ITEM(intlist, i);
        unsigned long x = PyLong_AsUnsignedLong(v);
        if (x == (unsigned long) -1 && PyErr_Occurred())
            return NULL;
        state->mt[i] = (unsigned int) x;
    }
    state->has_gauss = 0;
    state->gauss = 0.0;
    Py_RETURN_NONE;
}

NUMBA_EXPORT_FUNC(PyObject *)
_numba_rnd_get_state(PyObject *self, PyObject *arg)
{
    PyObject *intlist;
    int i;
    rnd_state_t *state;
    if (!rnd_state_converter(arg, &state))
        return NULL;

    intlist = PyList_New(MT_N);
    if (intlist == NULL)
        return NULL;
    for (i = 0; i < MT_N; i++) {
        PyObject *v = PyLong_FromUnsignedLong(state->mt[i]);
        if (v == NULL) {
            Py_DECREF(intlist);
            return NULL;
        }
        PyList_SET_ITEM(intlist, i, v);
    }
    return Py_BuildValue("iN", state->index, intlist);
}

NUMBA_EXPORT_FUNC(PyObject *)
_numba_rnd_seed_with_urandom(PyObject *self, PyObject *args)
{
    rnd_state_t *state;
    Py_buffer buf;
    unsigned int *keys;
    unsigned char *bytes;
    size_t i, nkeys;

    if (!PyArg_ParseTuple(args, "O&s*:rnd_seed",
                          rnd_state_converter, &state, &buf)) {
        return NULL;
    }
    /* Make a copy to avoid alignment issues */
    nkeys = buf.len / sizeof(unsigned int);
    keys = (unsigned int *) PyMem_Malloc(nkeys * sizeof(unsigned int));
    if (keys == NULL) {
        PyBuffer_Release(&buf);
        return NULL;
    }
    bytes = (unsigned char *) buf.buf;
    for (i = 0; i < nkeys; i++, bytes += 4) {
        keys[i] = (bytes[3] << 24) + (bytes[2] << 16) +
                  (bytes[1] << 8) + (bytes[0] << 0);
    }
    PyBuffer_Release(&buf);
    rnd_init_by_array(state, keys, nkeys);
    PyMem_Free(keys);
    Py_RETURN_NONE;
}

NUMBA_EXPORT_FUNC(PyObject *)
_numba_rnd_seed(PyObject *self, PyObject *args)
{
    unsigned int seed;
    rnd_state_t *state;

    if (!PyArg_ParseTuple(args, "O&I:rnd_seed",
                          rnd_state_converter, &state, &seed)) {
        PyErr_Clear();
        return _numba_rnd_seed_with_urandom(self, args);
    }
    numba_rnd_init(state, seed);
    Py_RETURN_NONE;
}

/* Random distribution helpers.
 * Most code straight from Numpy's distributions.c. */

#ifndef M_PI
#define M_PI 3.14159265358979323846264338328
#endif

NUMBA_EXPORT_FUNC(unsigned int)
get_next_int32(rnd_state_t *state)
{
    unsigned int y;

    if (state->index == MT_N) {
        numba_rnd_shuffle(state);
        state->index = 0;
    }
    y = state->mt[state->index++];
    /* Tempering */
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680U;
    y ^= (y << 15) & 0xefc60000U;
    y ^= (y >> 18);
    return y;
}

NUMBA_EXPORT_FUNC(double)
get_next_double(rnd_state_t *state)
{
    double a = get_next_int32(state) >> 5;
    double b = get_next_int32(state) >> 6;
    return (a * 67108864.0 + b) / 9007199254740992.0;
}

NUMBA_EXPORT_FUNC(double)
loggam(double x)
{
    double x0, x2, xp, gl, gl0;
    long k, n;

    static double a[10] = {8.333333333333333e-02,-2.777777777777778e-03,
         7.936507936507937e-04,-5.952380952380952e-04,
         8.417508417508418e-04,-1.917526917526918e-03,
         6.410256410256410e-03,-2.955065359477124e-02,
         1.796443723688307e-01,-1.39243221690590e+00};
    x0 = x;
    n = 0;
    if ((x == 1.0) || (x == 2.0))
    {
        return 0.0;
    }
    else if (x <= 7.0)
    {
        n = (long)(7 - x);
        x0 = x + n;
    }
    x2 = 1.0/(x0*x0);
    xp = 2*M_PI;
    gl0 = a[9];
    for (k=8; k>=0; k--)
    {
        gl0 *= x2;
        gl0 += a[k];
    }
    gl = gl0/x0 + 0.5*log(xp) + (x0-0.5)*log(x0) - x0;
    if (x <= 7.0)
    {
        for (k=1; k<=n; k++)
        {
            gl -= log(x0-1.0);
            x0 -= 1.0;
        }
    }
    return gl;
}


NUMBA_EXPORT_FUNC(int64_t)
numba_poisson_ptrs(rnd_state_t *state, double lam)
{
    /* This method is invoked only if the parameter lambda of this
     * distribution is big enough ( >= 10 ). The algorithm used is
     * described in "Hörmann, W. 1992. 'The Transformed Rejection
     * Method for Generating Poisson Random Variables'.
     * The implementation comes straight from Numpy.
     */
    int64_t k;
    double U, V, slam, loglam, a, b, invalpha, vr, us;

    slam = sqrt(lam);
    loglam = log(lam);
    b = 0.931 + 2.53*slam;
    a = -0.059 + 0.02483*b;
    invalpha = 1.1239 + 1.1328/(b-3.4);
    vr = 0.9277 - 3.6224/(b-2);

    while (1)
    {
        U = get_next_double(state) - 0.5;
        V = get_next_double(state);
        us = 0.5 - fabs(U);
        k = (int64_t) floor((2*a/us + b)*U + lam + 0.43);
        if ((us >= 0.07) && (V <= vr))
        {
            return k;
        }
        if ((k < 0) ||
            ((us < 0.013) && (V > us)))
        {
            continue;
        }
        if ((log(V) + log(invalpha) - log(a/(us*us)+b)) <=
            (-lam + k*loglam - loggam(k+1)))
        {
            return k;
        }
    }
}

/*
 * Other helpers.
 */

/* provide 64-bit division function to 32-bit platforms */
NUMBA_EXPORT_FUNC(int64_t)
numba_sdiv(int64_t a, int64_t b) {
    return a / b;
}

NUMBA_EXPORT_FUNC(uint64_t)
numba_udiv(uint64_t a, uint64_t b) {
    return a / b;
}

/* provide 64-bit remainder function to 32-bit platforms */
NUMBA_EXPORT_FUNC(int64_t)
numba_srem(int64_t a, int64_t b) {
    return a % b;
}

NUMBA_EXPORT_FUNC(uint64_t)
numba_urem(uint64_t a, uint64_t b) {
    return a % b;
}

/* provide frexp and ldexp; these wrappers deal with special cases
 * (zero, nan, infinity) directly, to sidestep platform differences.
 */
NUMBA_EXPORT_FUNC(double)
numba_frexp(double x, int *exp)
{
    if (!Py_IS_FINITE(x) || !x)
        *exp = 0;
    else
        x = frexp(x, exp);
    return x;
}

NUMBA_EXPORT_FUNC(float)
numba_frexpf(float x, int *exp)
{
    if (Py_IS_NAN(x) || Py_IS_INFINITY(x) || !x)
        *exp = 0;
    else
        x = frexpf(x, exp);
    return x;
}

NUMBA_EXPORT_FUNC(double)
numba_ldexp(double x, int exp)
{
    if (Py_IS_FINITE(x) && x && exp)
        x = ldexp(x, exp);
    return x;
}

NUMBA_EXPORT_FUNC(float)
numba_ldexpf(float x, int exp)
{
    if (Py_IS_FINITE(x) && x && exp)
        x = ldexpf(x, exp);
    return x;
}

/* provide complex power */
NUMBA_EXPORT_FUNC(void)
numba_cpow(Py_complex *a, Py_complex *b, Py_complex *c) {
    *c = _Py_c_pow(*a, *b);
}

/* provide gamma() and lgamma(); code borrowed from CPython */

/*
   sin(pi*x), giving accurate results for all finite x (especially x
   integral or close to an integer).  This is here for use in the
   reflection formula for the gamma function.  It conforms to IEEE
   754-2008 for finite arguments, but not for infinities or nans.
*/

static const double pi = 3.141592653589793238462643383279502884197;
static const double sqrtpi = 1.772453850905516027298167483341145182798;
static const double logpi = 1.144729885849400174143427351353058711647;

static double
sinpi(double x)
{
    double y, r;
    int n;
    /* this function should only ever be called for finite arguments */
    assert(Py_IS_FINITE(x));
    y = fmod(fabs(x), 2.0);
    n = (int)round(2.0*y);
    assert(0 <= n && n <= 4);
    switch (n) {
    case 0:
        r = sin(pi*y);
        break;
    case 1:
        r = cos(pi*(y-0.5));
        break;
    case 2:
        /* N.B. -sin(pi*(y-1.0)) is *not* equivalent: it would give
           -0.0 instead of 0.0 when y == 1.0. */
        r = sin(pi*(1.0-y));
        break;
    case 3:
        r = -cos(pi*(y-1.5));
        break;
    case 4:
        r = sin(pi*(y-2.0));
        break;
    default:
        assert(0);  /* should never get here */
        r = -1.23e200; /* silence gcc warning */
    }
    return copysign(1.0, x)*r;
}

/* Implementation of the real gamma function.  In extensive but non-exhaustive
   random tests, this function proved accurate to within <= 10 ulps across the
   entire float domain.  Note that accuracy may depend on the quality of the
   system math functions, the pow function in particular.  Special cases
   follow C99 annex F.  The parameters and method are tailored to platforms
   whose double format is the IEEE 754 binary64 format.

   Method: for x > 0.0 we use the Lanczos approximation with parameters N=13
   and g=6.024680040776729583740234375; these parameters are amongst those
   used by the Boost library.  Following Boost (again), we re-express the
   Lanczos sum as a rational function, and compute it that way.  The
   coefficients below were computed independently using MPFR, and have been
   double-checked against the coefficients in the Boost source code.

   For x < 0.0 we use the reflection formula.

   There's one minor tweak that deserves explanation: Lanczos' formula for
   Gamma(x) involves computing pow(x+g-0.5, x-0.5) / exp(x+g-0.5).  For many x
   values, x+g-0.5 can be represented exactly.  However, in cases where it
   can't be represented exactly the small error in x+g-0.5 can be magnified
   significantly by the pow and exp calls, especially for large x.  A cheap
   correction is to multiply by (1 + e*g/(x+g-0.5)), where e is the error
   involved in the computation of x+g-0.5 (that is, e = computed value of
   x+g-0.5 - exact value of x+g-0.5).  Here's the proof:

   Correction factor
   -----------------
   Write x+g-0.5 = y-e, where y is exactly representable as an IEEE 754
   double, and e is tiny.  Then:

     pow(x+g-0.5,x-0.5)/exp(x+g-0.5) = pow(y-e, x-0.5)/exp(y-e)
     = pow(y, x-0.5)/exp(y) * C,

   where the correction_factor C is given by

     C = pow(1-e/y, x-0.5) * exp(e)

   Since e is tiny, pow(1-e/y, x-0.5) ~ 1-(x-0.5)*e/y, and exp(x) ~ 1+e, so:

     C ~ (1-(x-0.5)*e/y) * (1+e) ~ 1 + e*(y-(x-0.5))/y

   But y-(x-0.5) = g+e, and g+e ~ g.  So we get C ~ 1 + e*g/y, and

     pow(x+g-0.5,x-0.5)/exp(x+g-0.5) ~ pow(y, x-0.5)/exp(y) * (1 + e*g/y),

   Note that for accuracy, when computing r*C it's better to do

     r + e*g/y*r;

   than

     r * (1 + e*g/y);

   since the addition in the latter throws away most of the bits of
   information in e*g/y.
*/

#define LANCZOS_N 13
static const double lanczos_g = 6.024680040776729583740234375;
static const double lanczos_g_minus_half = 5.524680040776729583740234375;
static const double lanczos_num_coeffs[LANCZOS_N] = {
    23531376880.410759688572007674451636754734846804940,
    42919803642.649098768957899047001988850926355848959,
    35711959237.355668049440185451547166705960488635843,
    17921034426.037209699919755754458931112671403265390,
    6039542586.3520280050642916443072979210699388420708,
    1439720407.3117216736632230727949123939715485786772,
    248874557.86205415651146038641322942321632125127801,
    31426415.585400194380614231628318205362874684987640,
    2876370.6289353724412254090516208496135991145378768,
    186056.26539522349504029498971604569928220784236328,
    8071.6720023658162106380029022722506138218516325024,
    210.82427775157934587250973392071336271166969580291,
    2.5066282746310002701649081771338373386264310793408
};

/* denominator is x*(x+1)*...*(x+LANCZOS_N-2) */
static const double lanczos_den_coeffs[LANCZOS_N] = {
    0.0, 39916800.0, 120543840.0, 150917976.0, 105258076.0, 45995730.0,
    13339535.0, 2637558.0, 357423.0, 32670.0, 1925.0, 66.0, 1.0};

/* gamma values for small positive integers, 1 though NGAMMA_INTEGRAL */
#define NGAMMA_INTEGRAL 23
static const double gamma_integral[NGAMMA_INTEGRAL] = {
    1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0, 362880.0,
    3628800.0, 39916800.0, 479001600.0, 6227020800.0, 87178291200.0,
    1307674368000.0, 20922789888000.0, 355687428096000.0,
    6402373705728000.0, 121645100408832000.0, 2432902008176640000.0,
    51090942171709440000.0, 1124000727777607680000.0,
};

/* Lanczos' sum L_g(x), for positive x */

static double
lanczos_sum(double x)
{
    double num = 0.0, den = 0.0;
    int i;
    assert(x > 0.0);
    /* evaluate the rational function lanczos_sum(x).  For large
       x, the obvious algorithm risks overflow, so we instead
       rescale the denominator and numerator of the rational
       function by x**(1-LANCZOS_N) and treat this as a
       rational function in 1/x.  This also reduces the error for
       larger x values.  The choice of cutoff point (5.0 below) is
       somewhat arbitrary; in tests, smaller cutoff values than
       this resulted in lower accuracy. */
    if (x < 5.0) {
        for (i = LANCZOS_N; --i >= 0; ) {
            num = num * x + lanczos_num_coeffs[i];
            den = den * x + lanczos_den_coeffs[i];
        }
    }
    else {
        for (i = 0; i < LANCZOS_N; i++) {
            num = num / x + lanczos_num_coeffs[i];
            den = den / x + lanczos_den_coeffs[i];
        }
    }
    return num/den;
}

NUMBA_EXPORT_FUNC(double)
numba_gamma(double x)
{
    double absx, r, y, z, sqrtpow;

    /* special cases */
    if (!Py_IS_FINITE(x)) {
        if (Py_IS_NAN(x) || x > 0.0)
            return x;  /* tgamma(nan) = nan, tgamma(inf) = inf */
        else {
            /*errno = EDOM;*/
            return Py_NAN;  /* tgamma(-inf) = nan, invalid */
        }
    }
    if (x == 0.0) {
        /*errno = EDOM;*/
        /* tgamma(+-0.0) = +-inf, divide-by-zero */
        return copysign(Py_HUGE_VAL, x);
    }

    /* integer arguments */
    if (x == floor(x)) {
        if (x < 0.0) {
            /*errno = EDOM;*/  /* tgamma(n) = nan, invalid for */
            return Py_NAN; /* negative integers n */
        }
        if (x <= NGAMMA_INTEGRAL)
            return gamma_integral[(int)x - 1];
    }
    absx = fabs(x);

    /* tiny arguments:  tgamma(x) ~ 1/x for x near 0 */
    if (absx < 1e-20) {
        r = 1.0/x;
        /*if (Py_IS_INFINITY(r))
            errno = ERANGE;*/
        return r;
    }

    /* large arguments: assuming IEEE 754 doubles, tgamma(x) overflows for
       x > 200, and underflows to +-0.0 for x < -200, not a negative
       integer. */
    if (absx > 200.0) {
        if (x < 0.0) {
            return 0.0/sinpi(x);
        }
        else {
            /*errno = ERANGE;*/
            return Py_HUGE_VAL;
        }
    }

    y = absx + lanczos_g_minus_half;
    /* compute error in sum */
    if (absx > lanczos_g_minus_half) {
        /* note: the correction can be foiled by an optimizing
           compiler that (incorrectly) thinks that an expression like
           a + b - a - b can be optimized to 0.0.  This shouldn't
           happen in a standards-conforming compiler. */
        double q = y - absx;
        z = q - lanczos_g_minus_half;
    }
    else {
        double q = y - lanczos_g_minus_half;
        z = q - absx;
    }
    z = z * lanczos_g / y;
    if (x < 0.0) {
        r = -pi / sinpi(absx) / absx * exp(y) / lanczos_sum(absx);
        r -= z * r;
        if (absx < 140.0) {
            r /= pow(y, absx - 0.5);
        }
        else {
            sqrtpow = pow(y, absx / 2.0 - 0.25);
            r /= sqrtpow;
            r /= sqrtpow;
        }
    }
    else {
        r = lanczos_sum(absx) / exp(y);
        r += z * r;
        if (absx < 140.0) {
            r *= pow(y, absx - 0.5);
        }
        else {
            sqrtpow = pow(y, absx / 2.0 - 0.25);
            r *= sqrtpow;
            r *= sqrtpow;
        }
    }
    /*if (Py_IS_INFINITY(r))
        errno = ERANGE;*/
    return r;
}

NUMBA_EXPORT_FUNC(float)
numba_gammaf(float x)
{
    return (float) numba_gamma(x);
}

/*
   lgamma:  natural log of the absolute value of the Gamma function.
   For large arguments, Lanczos' formula works extremely well here.
*/

NUMBA_EXPORT_FUNC(double)
numba_lgamma(double x)
{
    double r, absx;

    /* special cases */
    if (!Py_IS_FINITE(x)) {
        if (Py_IS_NAN(x))
            return x;  /* lgamma(nan) = nan */
        else
            return Py_HUGE_VAL; /* lgamma(+-inf) = +inf */
    }

    /* integer arguments */
    if (x == floor(x) && x <= 2.0) {
        if (x <= 0.0) {
            /*errno = EDOM;*/  /* lgamma(n) = inf, divide-by-zero for */
            return Py_HUGE_VAL; /* integers n <= 0 */
        }
        else {
            return 0.0; /* lgamma(1) = lgamma(2) = 0.0 */
        }
    }

    absx = fabs(x);
    /* tiny arguments: lgamma(x) ~ -log(fabs(x)) for small x */
    if (absx < 1e-20)
        return -log(absx);

    /* Lanczos' formula.  We could save a fraction of a ulp in accuracy by
       having a second set of numerator coefficients for lanczos_sum that
       absorbed the exp(-lanczos_g) term, and throwing out the lanczos_g
       subtraction below; it's probably not worth it. */
    r = log(lanczos_sum(absx)) - lanczos_g;
    r += (absx - 0.5) * (log(absx + lanczos_g - 0.5) - 1);
    if (x < 0.0)
        /* Use reflection formula to get value for negative x. */
        r = logpi - log(fabs(sinpi(absx))) - log(absx) - r;
    /*if (Py_IS_INFINITY(r))
        errno = ERANGE;*/
    return r;
}

NUMBA_EXPORT_FUNC(float)
numba_lgammaf(float x)
{
    return (float) numba_lgamma(x);
}

/* provide erf() and erfc(); code borrowed from CPython */

/*
   Implementations of the error function erf(x) and the complementary error
   function erfc(x).

   Method: following 'Numerical Recipes' by Flannery, Press et. al. (2nd ed.,
   Cambridge University Press), we use a series approximation for erf for
   small x, and a continued fraction approximation for erfc(x) for larger x;
   combined with the relations erf(-x) = -erf(x) and erfc(x) = 1.0 - erf(x),
   this gives us erf(x) and erfc(x) for all x.

   The series expansion used is:

      erf(x) = x*exp(-x*x)/sqrt(pi) * [
                     2/1 + 4/3 x**2 + 8/15 x**4 + 16/105 x**6 + ...]

   The coefficient of x**(2k-2) here is 4**k*factorial(k)/factorial(2*k).
   This series converges well for smallish x, but slowly for larger x.

   The continued fraction expansion used is:

      erfc(x) = x*exp(-x*x)/sqrt(pi) * [1/(0.5 + x**2 -) 0.5/(2.5 + x**2 - )
                              3.0/(4.5 + x**2 - ) 7.5/(6.5 + x**2 - ) ...]

   after the first term, the general term has the form:

      k*(k-0.5)/(2*k+0.5 + x**2 - ...).

   This expansion converges fast for larger x, but convergence becomes
   infinitely slow as x approaches 0.0.  The (somewhat naive) continued
   fraction evaluation algorithm used below also risks overflow for large x;
   but for large x, erfc(x) == 0.0 to within machine precision.  (For
   example, erfc(30.0) is approximately 2.56e-393).

   Parameters: use series expansion for abs(x) < ERF_SERIES_CUTOFF and
   continued fraction expansion for ERF_SERIES_CUTOFF <= abs(x) <
   ERFC_CONTFRAC_CUTOFF.  ERFC_SERIES_TERMS and ERFC_CONTFRAC_TERMS are the
   numbers of terms to use for the relevant expansions.  */

#define ERF_SERIES_CUTOFF 1.5
#define ERF_SERIES_TERMS 25
#define ERFC_CONTFRAC_CUTOFF 30.0
#define ERFC_CONTFRAC_TERMS 50

/*
   Error function, via power series.

   Given a finite float x, return an approximation to erf(x).
   Converges reasonably fast for small x.
*/

static double
m_erf_series(double x)
{
    double x2, acc, fk, result;
    int i, saved_errno;

    x2 = x * x;
    acc = 0.0;
    fk = (double)ERF_SERIES_TERMS + 0.5;
    for (i = 0; i < ERF_SERIES_TERMS; i++) {
        acc = 2.0 + x2 * acc / fk;
        fk -= 1.0;
    }
    /* Make sure the exp call doesn't affect errno;
       see m_erfc_contfrac for more. */
    saved_errno = errno;
    result = acc * x * exp(-x2) / sqrtpi;
    errno = saved_errno;
    return result;
}

/*
   Complementary error function, via continued fraction expansion.

   Given a positive float x, return an approximation to erfc(x).  Converges
   reasonably fast for x large (say, x > 2.0), and should be safe from
   overflow if x and nterms are not too large.  On an IEEE 754 machine, with x
   <= 30.0, we're safe up to nterms = 100.  For x >= 30.0, erfc(x) is smaller
   than the smallest representable nonzero float.  */

static double
m_erfc_contfrac(double x)
{
    double x2, a, da, p, p_last, q, q_last, b, result;
    int i, saved_errno;

    if (x >= ERFC_CONTFRAC_CUTOFF)
        return 0.0;

    x2 = x*x;
    a = 0.0;
    da = 0.5;
    p = 1.0; p_last = 0.0;
    q = da + x2; q_last = 1.0;
    for (i = 0; i < ERFC_CONTFRAC_TERMS; i++) {
        double temp;
        a += da;
        da += 2.0;
        b = da + x2;
        temp = p; p = b*p - a*p_last; p_last = temp;
        temp = q; q = b*q - a*q_last; q_last = temp;
    }
    /* Issue #8986: On some platforms, exp sets errno on underflow to zero;
       save the current errno value so that we can restore it later. */
    saved_errno = errno;
    result = p / q * x * exp(-x2) / sqrtpi;
    errno = saved_errno;
    return result;
}

/* Error function erf(x), for general x */

NUMBA_EXPORT_FUNC(double)
numba_erf(double x)
{
    double absx, cf;

    if (Py_IS_NAN(x))
        return x;
    absx = fabs(x);
    if (absx < ERF_SERIES_CUTOFF)
        return m_erf_series(x);
    else {
        cf = m_erfc_contfrac(absx);
        return x > 0.0 ? 1.0 - cf : cf - 1.0;
    }
}

NUMBA_EXPORT_FUNC(float)
numba_erff(float x)
{
    return (float) numba_erf(x);
}

/* Complementary error function erfc(x), for general x. */

NUMBA_EXPORT_FUNC(double)
numba_erfc(double x)
{
    double absx, cf;

    if (Py_IS_NAN(x))
        return x;
    absx = fabs(x);
    if (absx < ERF_SERIES_CUTOFF)
        return 1.0 - m_erf_series(x);
    else {
        cf = m_erfc_contfrac(absx);
        return x > 0.0 ? cf : 2.0 - cf;
    }
}

NUMBA_EXPORT_FUNC(float)
numba_erfcf(float x)
{
    return (float) numba_erfc(x);
}


NUMBA_EXPORT_FUNC(int)
numba_complex_adaptor(PyObject* obj, Py_complex *out) {
    PyObject* fobj;
    PyArray_Descr *dtype;
    double val[2];

    // Convert from python complex or numpy complex128
    if (PyComplex_Check(obj)) {
        out->real = PyComplex_RealAsDouble(obj);
        out->imag = PyComplex_ImagAsDouble(obj);
    }
    // Convert from numpy complex64
    else if (PyArray_IsScalar(obj, ComplexFloating)) {
        dtype = PyArray_DescrFromScalar(obj);
        if (dtype == NULL) {
            return 0;
        }
        if (PyArray_CastScalarDirect(obj, dtype, &val[0], NPY_CDOUBLE) < 0) {
            Py_DECREF(dtype);
            return 0;
        }
        out->real = val[0];
        out->imag = val[1];
        Py_DECREF(dtype);
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
NUMBA_EXPORT_FUNC(void *)
numba_extract_record_data(PyObject *recordobj, Py_buffer *pbuf) {
    PyObject *attrdata;
    void *ptr;

    attrdata = PyObject_GetAttrString(recordobj, "data");
    if (!attrdata) return NULL;

    if (-1 == PyObject_GetBuffer(attrdata, pbuf, 0)){
        #if PY_MAJOR_VERSION >= 3
            Py_DECREF(attrdata);
            return NULL;
        #else
            /* HACK!!! */
            /* In Python 2.6, it will report no buffer interface for record
               even though it should */
            PyBufferObject_Hack *hack;

            /* Clear the error */
            PyErr_Clear();

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
                    Py_DECREF(attrdata);
                    return NULL;
                }
                ptr = (char*)ptr + hack->b_offset;
            }
        #endif
    } else {
        ptr = pbuf->buf;
    }
    Py_DECREF(attrdata);
    return ptr;
}

/*
 * Return a record instance with dtype as the record type, and backed
 * by a copy of the memory area pointed to by (pdata, size).
 */
NUMBA_EXPORT_FUNC(PyObject *)
numba_recreate_record(void *pdata, int size, PyObject *dtype) {
    PyObject *numpy = NULL;
    PyObject *numpy_record = NULL;
    PyObject *aryobj = NULL;
    PyObject *dtypearg = NULL;
    PyObject *record = NULL;
    PyArray_Descr *descr = NULL;

    numpy = PyImport_ImportModuleNoBlock("numpy");
    if (!numpy) goto CLEANUP;

    numpy_record = PyObject_GetAttrString(numpy, "record");
    if (!numpy_record) goto CLEANUP;

    dtypearg = PyTuple_Pack(2, numpy_record, dtype);
    if (!dtypearg || !PyArray_DescrConverter(dtypearg, &descr))
        goto CLEANUP;

    /* This steals a reference to descr, so we don't have to DECREF it */
    aryobj = PyArray_FromString(pdata, size, descr, 1, NULL);
    if (!aryobj) goto CLEANUP;

    record = PySequence_GetItem(aryobj, 0);

CLEANUP:
    Py_XDECREF(numpy);
    Py_XDECREF(numpy_record);
    Py_XDECREF(aryobj);
    Py_XDECREF(dtypearg);

    return record;
}

NUMBA_EXPORT_FUNC(int)
numba_adapt_ndarray(PyObject *obj, arystruct_t* arystruct) {
    PyArrayObject *ndary;
    int i, ndim;
    npy_intp *p;

    if (!PyArray_Check(obj)) {
        return -1;
    }

    ndary = (PyArrayObject*)obj;
    ndim = PyArray_NDIM(ndary);

    arystruct->data = PyArray_DATA(ndary);
    arystruct->nitems = PyArray_SIZE(ndary);
    arystruct->itemsize = PyArray_ITEMSIZE(ndary);
    arystruct->parent = obj;
    p = arystruct->shape_and_strides;
    for (i = 0; i < ndim; i++, p++) {
        *p = PyArray_DIM(ndary, i);
    }
    for (i = 0; i < ndim; i++, p++) {
        *p = PyArray_STRIDE(ndary, i);
    }
    arystruct->meminfo = NULL;
    return 0;
}

NUMBA_EXPORT_FUNC(int)
numba_get_buffer(PyObject *obj, Py_buffer *buf)
{
    /* Ask for shape and strides, but no suboffsets */
    return PyObject_GetBuffer(obj, buf, PyBUF_RECORDS_RO);
}

NUMBA_EXPORT_FUNC(void)
numba_adapt_buffer(Py_buffer *buf, arystruct_t *arystruct)
{
    int i;
    npy_intp *p;

    arystruct->data = buf->buf;
    arystruct->itemsize = buf->itemsize;
    arystruct->parent = buf->obj;
    arystruct->nitems = 1;
    p = arystruct->shape_and_strides;
    for (i = 0; i < buf->ndim; i++, p++) {
        *p = buf->shape[i];
        arystruct->nitems *= buf->shape[i];
    }
    for (i = 0; i < buf->ndim; i++, p++) {
        *p = buf->strides[i];
    }
    arystruct->meminfo = NULL;
}

NUMBA_EXPORT_FUNC(void)
numba_release_buffer(Py_buffer *buf)
{
    PyBuffer_Release(buf);
}

NUMBA_EXPORT_FUNC(PyObject *)
numba_ndarray_new(int nd,
                  npy_intp *dims,   /* shape */
                  npy_intp *strides,
                  void* data,
                  int type_num,
                  int itemsize)
{
    PyObject *ndary;
    int flags = NPY_ARRAY_BEHAVED;
    ndary = PyArray_New((PyTypeObject*)&PyArray_Type, nd, dims, type_num,
                       strides, data, 0, flags, NULL);
    return ndary;
}

/*
 * Straight from Numpy's _attempt_nocopy_reshape()
 * (np/core/src/multiarray/shape.c).
 * Attempt to reshape an array without copying data
 *
 * This function should correctly handle all reshapes, including
 * axes of length 1. Zero strides should work but are untested.
 *
 * If a copy is needed, returns 0
 * If no copy is needed, returns 1 and fills `npy_intp *newstrides`
 *     with appropriate strides
 */
NUMBA_EXPORT_FUNC(int)
numba_attempt_nocopy_reshape(npy_intp nd, const npy_intp *dims, const npy_intp *strides,
                             npy_intp newnd, const npy_intp *newdims,
                             npy_intp *newstrides, npy_intp itemsize,
                             int is_f_order)
{
    int oldnd;
    npy_intp olddims[NPY_MAXDIMS];
    npy_intp oldstrides[NPY_MAXDIMS];
    npy_intp np, op, last_stride;
    int oi, oj, ok, ni, nj, nk;

    oldnd = 0;
    /*
     * Remove axes with dimension 1 from the old array. They have no effect
     * but would need special cases since their strides do not matter.
     */
    for (oi = 0; oi < nd; oi++) {
        if (dims[oi]!= 1) {
            olddims[oldnd] = dims[oi];
            oldstrides[oldnd] = strides[oi];
            oldnd++;
        }
    }

    np = 1;
    for (ni = 0; ni < newnd; ni++) {
        np *= newdims[ni];
    }
    op = 1;
    for (oi = 0; oi < oldnd; oi++) {
        op *= olddims[oi];
    }
    if (np != op) {
        /* different total sizes; no hope */
        return 0;
    }

    if (np == 0) {
        /* the current code does not handle 0-sized arrays, so give up */
        return 0;
    }

    /* oi to oj and ni to nj give the axis ranges currently worked with */
    oi = 0;
    oj = 1;
    ni = 0;
    nj = 1;
    while (ni < newnd && oi < oldnd) {
        np = newdims[ni];
        op = olddims[oi];

        while (np != op) {
            if (np < op) {
                /* Misses trailing 1s, these are handled later */
                np *= newdims[nj++];
            } else {
                op *= olddims[oj++];
            }
        }

        /* Check whether the original axes can be combined */
        for (ok = oi; ok < oj - 1; ok++) {
            if (is_f_order) {
                if (oldstrides[ok+1] != olddims[ok]*oldstrides[ok]) {
                     /* not contiguous enough */
                    return 0;
                }
            }
            else {
                /* C order */
                if (oldstrides[ok] != olddims[ok+1]*oldstrides[ok+1]) {
                    /* not contiguous enough */
                    return 0;
                }
            }
        }

        /* Calculate new strides for all axes currently worked with */
        if (is_f_order) {
            newstrides[ni] = oldstrides[oi];
            for (nk = ni + 1; nk < nj; nk++) {
                newstrides[nk] = newstrides[nk - 1]*newdims[nk - 1];
            }
        }
        else {
            /* C order */
            newstrides[nj - 1] = oldstrides[oj - 1];
            for (nk = nj - 1; nk > ni; nk--) {
                newstrides[nk - 1] = newstrides[nk]*newdims[nk];
            }
        }
        ni = nj++;
        oi = oj++;
    }

    /*
     * Set strides corresponding to trailing 1s of the new shape.
     */
    if (ni >= 1) {
        last_stride = newstrides[ni - 1];
    }
    else {
        last_stride = itemsize;
    }
    if (is_f_order) {
        last_stride *= newdims[ni - 1];
    }
    for (nk = ni; nk < newnd; nk++) {
        newstrides[nk] = last_stride;
    }

    return 1;
}

/*
 * BLAS calling helpers.  The helpers can be called without the GIL held.
 * The caller is responsible for checking arguments (especially dimensions).
 */

/* Fetch the address of the given CBLAS function, as exposed by
   scipy.linalg.cython_blas. */
static void *
import_cblas_function(const char *function_name)
{
    PyObject *module, *capi, *cobj;
    void *res = NULL;

    module = PyImport_ImportModule("scipy.linalg.cython_blas");
    if (module == NULL)
        return NULL;
    capi = PyObject_GetAttrString(module, "__pyx_capi__");
    Py_DECREF(module);
    if (capi == NULL)
        return NULL;
    cobj = PyMapping_GetItemString(capi, function_name);
    Py_DECREF(capi);
    if (cobj == NULL)
        return NULL;
#if PY_MAJOR_VERSION >= 3 || (PY_MINOR_VERSION >= 7)
    {
        /* 2.7+ => Cython exports a PyCapsule */
        const char *capsule_name = PyCapsule_GetName(cobj);
        if (capsule_name != NULL) {
            res = PyCapsule_GetPointer(cobj, capsule_name);
        }
        Py_DECREF(cobj);
    }
#else
    {
        /* 2.6 => Cython exports a legacy PyCObject */
        res = PyCObject_AsVoidPtr(cobj);
        Py_DECREF(cobj);
    }
#endif
    return res;
}

/* Fast getters caching the value of a function's address after
   the first call to import_cblas_function(). */

#define EMIT_GET_CBLAS_FUNC(name)                            \
    static void *cblas_ ## name = NULL;                      \
    static void *get_cblas_ ## name(void) {                  \
        if (cblas_ ## name == NULL) {                        \
            PyGILState_STATE st = PyGILState_Ensure();       \
            cblas_ ## name = import_cblas_function(# name);  \
            PyGILState_Release(st);                          \
        }                                                    \
        return cblas_ ## name;                               \
    }

EMIT_GET_CBLAS_FUNC(dgemm)
EMIT_GET_CBLAS_FUNC(sgemm)
EMIT_GET_CBLAS_FUNC(cgemm)
EMIT_GET_CBLAS_FUNC(zgemm)
EMIT_GET_CBLAS_FUNC(dgemv)
EMIT_GET_CBLAS_FUNC(sgemv)
EMIT_GET_CBLAS_FUNC(cgemv)
EMIT_GET_CBLAS_FUNC(zgemv)
EMIT_GET_CBLAS_FUNC(ddot)
EMIT_GET_CBLAS_FUNC(sdot)
EMIT_GET_CBLAS_FUNC(cdotu)
EMIT_GET_CBLAS_FUNC(zdotu)
EMIT_GET_CBLAS_FUNC(cdotc)
EMIT_GET_CBLAS_FUNC(zdotc)

#undef EMIT_GET_CBLAS_FUNC

typedef float (*sdot_t)(int *n, void *dx, int *incx, void *dy, int *incy);
typedef double (*ddot_t)(int *n, void *dx, int *incx, void *dy, int *incy);
typedef npy_complex64 (*cdot_t)(int *n, void *dx, int *incx, void *dy, int *incy);
typedef npy_complex128 (*zdot_t)(int *n, void *dx, int *incx, void *dy, int *incy);

typedef void (*xxgemv_t)(char *trans, int *m, int *n,
                         void *alpha, void *a, int *lda,
                         void *x, int *incx, void *beta,
                         void *y, int *incy);

typedef void (*xxgemm_t)(char *transa, char *transb,
                         int *m, int *n, int *k,
                         void *alpha, void *a, int *lda,
                         void *b, int *ldb, void *beta,
                         void *c, int *ldc);

/* Vector * vector: result = dx * dy */
NUMBA_EXPORT_FUNC(int)
numba_xxdot(char kind, char conjugate, Py_ssize_t n, void *dx, void *dy,
            void *result)
{
    void *raw_func = NULL;
    int _n;
    int inc = 1;

    switch (kind) {
        case 'd':
            raw_func = get_cblas_ddot();
            break;
        case 's':
            raw_func = get_cblas_sdot();
            break;
        case 'c':
            raw_func = conjugate ? get_cblas_cdotc() : get_cblas_cdotu();
            break;
        case 'z':
            raw_func = conjugate ? get_cblas_zdotc() : get_cblas_zdotu();
            break;
        default:
            {
                PyGILState_STATE st = PyGILState_Ensure();
                PyErr_SetString(PyExc_ValueError,
                                "invalid kind of *DOT function");
                PyGILState_Release(st);
            }
            return -1;
    }
    if (raw_func == NULL)
        return -1;

    _n = (int) n;

    switch (kind) {
        case 'd':
            *(double *) result = (*(ddot_t) raw_func)(&_n, dx, &inc, dy, &inc);;
            break;
        case 's':
            *(float *) result = (*(sdot_t) raw_func)(&_n, dx, &inc, dy, &inc);;
            break;
        case 'c':
            *(npy_complex64 *) result = (*(cdot_t) raw_func)(&_n, dx, &inc, dy, &inc);;
            break;
        case 'z':
            *(npy_complex128 *) result = (*(zdot_t) raw_func)(&_n, dx, &inc, dy, &inc);;
            break;
    }

    return 0;
}

/* Matrix * vector: y = alpha * a * x + beta * y */
NUMBA_EXPORT_FUNC(int)
numba_xxgemv(char kind, char *trans, Py_ssize_t m, Py_ssize_t n,
             void *alpha, void *a, Py_ssize_t lda,
             void *x, void *beta, void *y)
{
    void *raw_func = NULL;
    int _m, _n;
    int _lda;
    int inc = 1;

    switch (kind) {
        case 'd':
            raw_func = get_cblas_dgemv();
            break;
        case 's':
            raw_func = get_cblas_sgemv();
            break;
        case 'c':
            raw_func = get_cblas_cgemv();
            break;
        case 'z':
            raw_func = get_cblas_zgemv();
            break;
        default:
            {
                PyGILState_STATE st = PyGILState_Ensure();
                PyErr_SetString(PyExc_ValueError,
                                "invalid kind of *GEMV function");
                PyGILState_Release(st);
            }
            return -1;
    }
    if (raw_func == NULL)
        return -1;

    _m = (int) m;
    _n = (int) n;
    _lda = (int) lda;

    (*(xxgemv_t) raw_func)(trans, &_m, &_n, alpha, a, &_lda,
                           x, &inc, beta, y, &inc);
    return 0;
}

/* Matrix * matrix: c = alpha * a * b + beta * c */
NUMBA_EXPORT_FUNC(int)
numba_xxgemm(char kind, char *transa, char *transb,
             Py_ssize_t m, Py_ssize_t n, Py_ssize_t k,
             void *alpha, void *a, Py_ssize_t lda,
             void *b, Py_ssize_t ldb, void *beta,
             void *c, Py_ssize_t ldc)
{
    void *raw_func = NULL;
    int _m, _n, _k;
    int _lda, _ldb, _ldc;

    switch (kind) {
        case 'd':
            raw_func = get_cblas_dgemm();
            break;
        case 's':
            raw_func = get_cblas_sgemm();
            break;
        case 'c':
            raw_func = get_cblas_cgemm();
            break;
        case 'z':
            raw_func = get_cblas_zgemm();
            break;
        default:
            {
                PyGILState_STATE st = PyGILState_Ensure();
                PyErr_SetString(PyExc_ValueError,
                                "invalid kind of *GEMM function");
                PyGILState_Release(st);
            }
            return -1;
    }
    if (raw_func == NULL)
        return -1;

    _m = (int) m;
    _n = (int) n;
    _k = (int) k;
    _lda = (int) lda;
    _ldb = (int) ldb;
    _ldc = (int) ldc;

    (*(xxgemm_t) raw_func)(transa, transb, &_m, &_n, &_k, alpha, a, &_lda,
                           b, &_ldb, beta, c, &_ldc);
    return 0;
}


/* We use separate functions for datetime64 and timedelta64, to ensure
 * proper type checking.
 */
NUMBA_EXPORT_FUNC(npy_int64)
numba_extract_np_datetime(PyObject *td)
{
    if (!PyArray_IsScalar(td, Datetime)) {
        PyErr_SetString(PyExc_TypeError,
                        "expected a numpy.datetime64 object");
        return -1;
    }
    return PyArrayScalar_VAL(td, Timedelta);
}

NUMBA_EXPORT_FUNC(npy_int64)
numba_extract_np_timedelta(PyObject *td)
{
    if (!PyArray_IsScalar(td, Timedelta)) {
        PyErr_SetString(PyExc_TypeError,
                        "expected a numpy.timedelta64 object");
        return -1;
    }
    return PyArrayScalar_VAL(td, Timedelta);
}

NUMBA_EXPORT_FUNC(PyObject *)
numba_create_np_datetime(npy_int64 value, int unit_code)
{
    PyDatetimeScalarObject *obj = (PyDatetimeScalarObject *)
        PyArrayScalar_New(Datetime);
    if (obj != NULL) {
        obj->obval = value;
        obj->obmeta.base = unit_code;
        obj->obmeta.num = 1;
    }
    return (PyObject *) obj;
}

NUMBA_EXPORT_FUNC(PyObject *)
numba_create_np_timedelta(npy_int64 value, int unit_code)
{
    PyTimedeltaScalarObject *obj = (PyTimedeltaScalarObject *)
        PyArrayScalar_New(Timedelta);
    if (obj != NULL) {
        obj->obval = value;
        obj->obmeta.base = unit_code;
        obj->obmeta.num = 1;
    }
    return (PyObject *) obj;
}

NUMBA_EXPORT_FUNC(uint64_t)
numba_fptoui(double x) {
    /* First cast to signed int of the full width to make sure sign extension
       happens (this can make a difference on some platforms...). */
    return (uint64_t) (int64_t) x;
}

NUMBA_EXPORT_FUNC(uint64_t)
numba_fptouif(float x) {
    return (uint64_t) (int64_t) x;
}

NUMBA_EXPORT_FUNC(void)
numba_gil_ensure(PyGILState_STATE *state) {
    *state = PyGILState_Ensure();
}

NUMBA_EXPORT_FUNC(void)
numba_gil_release(PyGILState_STATE *state) {
    PyGILState_Release(*state);
}

/* Pointer-stuffing functions for tagging a Python list object with an
 * arbitrary pointer.
 * Note a similar hack is used by Python itself, since
 * "list.sort() temporarily sets allocated to -1 to detect mutations".
 */

NUMBA_EXPORT_FUNC(void)
numba_set_list_private_data(PyListObject *listobj, void *ptr)
{
    /* Since ptr is dynamically allocated, it is at least
     * 4- or 8-byte-aligned, meaning we can shift it by a couple bits
     * to the right without losing information.
     */
    if ((size_t) ptr & 1) {
        /* Should never happen */
        Py_FatalError("Numba_set_list_private_data got an unaligned pointer");
    }
    /* Make the pointer distinguishable by forcing it into a negative
     * number (obj->allocated is normally positive, except when sorting
     * where it's changed to -1).
     */
    listobj->allocated = - (Py_ssize_t) ((size_t) ptr >> 1);
}

NUMBA_EXPORT_FUNC(void *)
numba_get_list_private_data(PyListObject *listobj)
{
    if (listobj->allocated < -1) {
        /* A Numba pointer is stuffed in the list, return it */
        return (void *) ((size_t) -listobj->allocated << 1);
    }
    return NULL;
}

NUMBA_EXPORT_FUNC(void)
numba_reset_list_private_data(PyListObject *listobj)
{
    /* Pretend there is no over-allocation; this should be always correct,
     * if not optimal.
     */
    if (listobj->allocated < -1)
        listobj->allocated = PyList_GET_SIZE(listobj);
}

NUMBA_EXPORT_FUNC(int)
numba_unpack_slice(PyObject *obj,
                   Py_ssize_t *start, Py_ssize_t *stop, Py_ssize_t *step)
{
    PySliceObject *slice = (PySliceObject *) obj;
    if (!PySlice_Check(obj)) {
        PyErr_Format(PyExc_TypeError,
                     "Expected a slice object, got '%s'",
                     Py_TYPE(slice)->tp_name);
        return -1;
    }
#define FETCH_MEMBER(NAME)                                      \
    if (slice->NAME != Py_None) {                               \
        Py_ssize_t v = PyNumber_AsSsize_t(slice->NAME,          \
                                          PyExc_OverflowError); \
        if (v == -1 && PyErr_Occurred())                        \
            return -1;                                          \
        *NAME = v;                                              \
    }
    FETCH_MEMBER(start)
    FETCH_MEMBER(stop)
    FETCH_MEMBER(step)
    return 0;

#undef FETCH_MEMBER
}

/* Logic for raising an arbitrary object.  Adapted from CPython's ceval.c.
   This *consumes* a reference count to its argument. */
NUMBA_EXPORT_FUNC(int)
numba_do_raise(PyObject *exc)
{
    PyObject *type = NULL, *value = NULL;

    /* We support the following forms of raise:
       raise
       raise <instance>
       raise <type> */

    if (exc == Py_None) {
        /* Reraise */
        PyThreadState *tstate = PyThreadState_GET();
        PyObject *tb;
        Py_DECREF(exc);
        type = tstate->exc_type;
        value = tstate->exc_value;
        tb = tstate->exc_traceback;
        if (type == Py_None) {
            PyErr_SetString(PyExc_RuntimeError,
                            "No active exception to reraise");
            return 0;
        }
        Py_XINCREF(type);
        Py_XINCREF(value);
        Py_XINCREF(tb);
        PyErr_Restore(type, value, tb);
        return 1;
    }

    if (PyTuple_CheckExact(exc)) {
        /* A (callable, arguments) tuple. */
        if (!PyArg_ParseTuple(exc, "OO", &type, &value)) {
            Py_DECREF(exc);
            goto raise_error;
        }
        value = PyObject_CallObject(type, value);
        Py_DECREF(exc);
        type = NULL;
        if (value == NULL)
            goto raise_error;
        if (!PyExceptionInstance_Check(value)) {
            PyErr_SetString(PyExc_TypeError,
                            "exceptions must derive from BaseException");
            goto raise_error;
        }
        type = PyExceptionInstance_Class(value);
        Py_INCREF(type);
    }
    else if (PyExceptionClass_Check(exc)) {
        type = exc;
        value = PyObject_CallObject(exc, NULL);
        if (value == NULL)
            goto raise_error;
        if (!PyExceptionInstance_Check(value)) {
            PyErr_SetString(PyExc_TypeError,
                            "exceptions must derive from BaseException");
            goto raise_error;
        }
    }
    else if (PyExceptionInstance_Check(exc)) {
        value = exc;
        type = PyExceptionInstance_Class(exc);
        Py_INCREF(type);
    }
    else {
        /* Not something you can raise.  You get an exception
           anyway, just not what you specified :-) */
        Py_DECREF(exc);
        PyErr_SetString(PyExc_TypeError,
                        "exceptions must derive from BaseException");
        goto raise_error;
    }

    PyErr_SetObject(type, value);
    /* PyErr_SetObject incref's its arguments */
    Py_XDECREF(value);
    Py_XDECREF(type);
    return 0;

raise_error:
    Py_XDECREF(value);
    Py_XDECREF(type);
    return 0;
}

NUMBA_EXPORT_FUNC(PyObject *)
numba_unpickle(const char *data, int n)
{
    PyObject *buf, *obj;
    static PyObject *loads;

    /* Caching the pickle.loads function shaves a couple µs here. */
    if (loads == NULL) {
        PyObject *picklemod;
#if PY_MAJOR_VERSION >= 3
        picklemod = PyImport_ImportModule("pickle");
#else
        picklemod = PyImport_ImportModule("cPickle");
#endif
        if (picklemod == NULL)
            return NULL;
        loads = PyObject_GetAttrString(picklemod, "loads");
        Py_DECREF(picklemod);
        if (loads == NULL)
            return NULL;
    }

    buf = PyBytes_FromStringAndSize(data, n);
    if (buf == NULL)
        return NULL;
    obj = PyObject_CallFunctionObjArgs(loads, buf, NULL);
    Py_DECREF(buf);
    return obj;
}


/*
Define bridge for all math functions
*/

#define MATH_UNARY(F, R, A) \
    NUMBA_EXPORT_FUNC(R) numba_##F(A a) { return F(a); }
#define MATH_BINARY(F, R, A, B) \
    NUMBA_EXPORT_FUNC(R) numba_##F(A a, B b) { return F(a, b); }

#include "mathnames.h"

#undef MATH_UNARY
#undef MATH_BINARY
