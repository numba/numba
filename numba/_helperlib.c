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

static rnd_state_t py_random_state;
static rnd_state_t np_random_state;

/* Some code portions below from CPython's _randommodule.c, some others
   from Numpy's and Jean-Sebastien Roy's randomkit.c. */

static void
Numba_rnd_shuffle(rnd_state_t *state)
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
static void
Numba_rnd_init(rnd_state_t *state, unsigned int seed)
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

    Numba_rnd_init(state, 19650218U);
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
static int
_rnd_random_seed(rnd_state_t *state)
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
    Numba_rnd_init(state, seed);
    return 0;
}

/* Python-exposed helpers for state management */
static int
rnd_state_converter(PyObject *obj, rnd_state_t **state)
{
    *state = (rnd_state_t *) PyLong_AsVoidPtr(obj);
    return (*state != NULL || !PyErr_Occurred());
}

static PyObject *
rnd_shuffle(PyObject *self, PyObject *arg)
{
    rnd_state_t *state;
    if (!rnd_state_converter(arg, &state))
        return NULL;
    Numba_rnd_shuffle(state);
    Py_RETURN_NONE;
}

static PyObject *
rnd_set_state(PyObject *self, PyObject *args)
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

static PyObject *
rnd_get_state(PyObject *self, PyObject *arg)
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

static PyObject *
rnd_seed_with_urandom(PyObject *self, PyObject *args)
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

static PyObject *
rnd_seed(PyObject *self, PyObject *args)
{
    unsigned int seed;
    rnd_state_t *state;

    if (!PyArg_ParseTuple(args, "O&I:rnd_seed",
                          rnd_state_converter, &state, &seed)) {
        PyErr_Clear();
        return rnd_seed_with_urandom(self, args);
    }
    Numba_rnd_init(state, seed);
    Py_RETURN_NONE;
}

/* Random distribution helpers.
 * Most code straight from Numpy's distributions.c. */

#ifndef M_PI
#define M_PI 3.14159265358979323846264338328
#endif

static unsigned int
get_next_int32(rnd_state_t *state)
{
    unsigned int y;

    if (state->index == MT_N) {
        Numba_rnd_shuffle(state);
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

static double
get_next_double(rnd_state_t *state)
{
    double a = get_next_int32(state) >> 5;
    double b = get_next_int32(state) >> 6;
    return (a * 67108864.0 + b) / 9007199254740992.0;
}

static double
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


static int64_t
Numba_poisson_ptrs(rnd_state_t *state, double lam)
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

/* provide 128-bit multiplilcation inplace for __multi3
 * on 32-bit platform */

typedef union i128 {
    struct {
        uint64_t low;
        int64_t high;
    } s;
} i128;


static
i128 umul64(uint64_t a, uint64_t b) {
    /* Adapted from __mulddi in compiler-rt */
    i128 r;
    uint64_t t;
    const int bits_in_dword_2 = 32;
    const uint64_t lower_mask = (uint64_t)~0 >> bits_in_dword_2;

    r.s.low = (a & lower_mask) * (b & lower_mask);
    t = r.s.low >> bits_in_dword_2;
    r.s.low &= lower_mask;
    t += (a >> bits_in_dword_2) * (b & lower_mask);
    r.s.low += (t & lower_mask) << bits_in_dword_2;
    r.s.high = t >> bits_in_dword_2;
    t = r.s.low >> bits_in_dword_2;
    r.s.low &= lower_mask;
    t += (b >> bits_in_dword_2) * (a & lower_mask);
    r.s.low += (t & lower_mask) << bits_in_dword_2;
    r.s.high += t >> bits_in_dword_2;
    r.s.high += (a >> bits_in_dword_2) * (b >> bits_in_dword_2);
    return r;
}

static
i128 mul128(i128 a, i128 b) {
    /* Adapted from __multi3 in compiler-rt */
    i128 r = umul64(a.s.low, b.s.low);
    r.s.high += a.s.high * b.s.low + a.s.low * b.s.high;
    return r;
}


static
i128
Numba_multi3(i128 x, i128 y) {
	return mul128(x, y);
}


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

/* provide frexp and ldexp; these wrappers deal with special cases
 * (zero, nan, infinity) directly, to sidestep platform differences.
 */
static
double Numba_frexp(double x, int *exp)
{
    if (!Py_IS_FINITE(x) || !x)
        *exp = 0;
    else
        x = frexp(x, exp);
    return x;
}

static
float Numba_frexpf(float x, int *exp)
{
    if (Py_IS_NAN(x) || Py_IS_INFINITY(x) || !x)
        *exp = 0;
    else
        x = frexpf(x, exp);
    return x;
}

static
double Numba_ldexp(double x, int exp)
{
    if (Py_IS_FINITE(x) && x && exp)
        x = ldexp(x, exp);
    return x;
}

static
float Numba_ldexpf(float x, int exp)
{
    if (Py_IS_FINITE(x) && x && exp)
        x = ldexpf(x, exp);
    return x;
}

/* provide complex power */
static
void Numba_cpow(Py_complex *a, Py_complex *b, Py_complex *c) {
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

static double
Numba_gamma(double x)
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

static float
Numba_gammaf(float x)
{
    return (float) Numba_gamma(x);
}

/*
   lgamma:  natural log of the absolute value of the Gamma function.
   For large arguments, Lanczos' formula works extremely well here.
*/

static double
Numba_lgamma(double x)
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

static float
Numba_lgammaf(float x)
{
    return (float) Numba_lgamma(x);
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

static double
Numba_erf(double x)
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

static float
Numba_erff(float x)
{
    return (float) Numba_erf(x);
}

/* Complementary error function erfc(x), for general x. */

static double
Numba_erfc(double x)
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

static float
Numba_erfcf(float x)
{
    return (float) Numba_erfc(x);
}


static
int Numba_complex_adaptor(PyObject* obj, Py_complex *out) {
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
static
void* Numba_extract_record_data(PyObject *recordobj, Py_buffer *pbuf) {
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
static
PyObject* Numba_recreate_record(void *pdata, int size, PyObject *dtype) {
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

static
int Numba_adapt_ndarray(PyObject *obj, arystruct_t* arystruct) {
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

static int
Numba_get_buffer(PyObject *obj, Py_buffer *buf)
{
    /* Ask for shape and strides, but no suboffsets */
    return PyObject_GetBuffer(obj, buf, PyBUF_RECORDS_RO);
}

static void
Numba_adapt_buffer(Py_buffer *buf, arystruct_t *arystruct)
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

static void
Numba_release_buffer(Py_buffer *buf)
{
    PyBuffer_Release(buf);
}

static
PyObject* Numba_ndarray_new(int nd,
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

/* We use separate functions for datetime64 and timedelta64, to ensure
 * proper type checking.
 */
static npy_int64
Numba_extract_np_datetime(PyObject *td)
{
    if (!PyArray_IsScalar(td, Datetime)) {
        PyErr_SetString(PyExc_TypeError,
                        "expected a numpy.datetime64 object");
        return -1;
    }
    return PyArrayScalar_VAL(td, Timedelta);
}

static npy_int64
Numba_extract_np_timedelta(PyObject *td)
{
    if (!PyArray_IsScalar(td, Timedelta)) {
        PyErr_SetString(PyExc_TypeError,
                        "expected a numpy.timedelta64 object");
        return -1;
    }
    return PyArrayScalar_VAL(td, Timedelta);
}

static PyObject *
Numba_create_np_datetime(npy_int64 value, int unit_code)
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

static PyObject *
Numba_create_np_timedelta(npy_int64 value, int unit_code)
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

static
uint64_t Numba_fptoui(double x) {
    /* First cast to signed int of the full width to make sure sign extension
       happens (this can make a difference on some platforms...). */
    return (uint64_t) (int64_t) x;
}

static
uint64_t Numba_fptouif(float x) {
    return (uint64_t) (int64_t) x;
}

static
void Numba_gil_ensure(PyGILState_STATE *state) {
    *state = PyGILState_Ensure();
}

static
void Numba_gil_release(PyGILState_STATE *state) {
    PyGILState_Release(*state);
}

/* Logic for raising an arbitrary object.  Adapted from CPython's ceval.c.
   This *consumes* a reference count to its argument. */
static int
Numba_do_raise(PyObject *exc)
{
    PyObject *type = NULL, *value = NULL;

    /* We support the following forms of raise:
       raise
       raise <instance>
       raise <type> */

    if (exc == NULL) {
        /* Reraise */
        PyThreadState *tstate = PyThreadState_GET();
        PyObject *tb;
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

static PyObject *
Numba_unpickle(const char *data, Py_ssize_t n)
{
    PyObject *buf, *picklemod, *obj;
    buf = PyBytes_FromStringAndSize(data, n);
    if (buf == NULL)
        return NULL;
#if PY_MAJOR_VERSION >= 3
    picklemod = PyImport_ImportModule("pickle");
#else
    picklemod = PyImport_ImportModule("cPickle");
#endif
    if (picklemod == NULL) {
        Py_DECREF(buf);
        return NULL;
    }
    obj = PyObject_CallMethod(picklemod, "loads", "O", buf);
    Py_DECREF(buf);
    Py_DECREF(picklemod);
    return obj;
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
Expose all functions
*/

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

#define declmethod(func) _declpointer(#func, &Numba_##func)

#define declpointer(ptr) _declpointer(#ptr, &ptr)
	declmethod(multi3);
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
    declmethod(do_raise);
    declmethod(unpickle);
    declmethod(rnd_shuffle);
    declmethod(rnd_init);
    declmethod(poisson_ptrs);

    declpointer(py_random_state);
    declpointer(np_random_state);

#define MATH_UNARY(F, R, A) declmethod(F);
#define MATH_BINARY(F, R, A, B) declmethod(F);
    #include "mathnames.inc"
#undef MATH_UNARY
#undef MATH_BINARY

#undef declmethod
    return dct;
error:
    Py_XDECREF(dct);
    return NULL;
}

static PyMethodDef ext_methods[] = {
    { "rnd_get_state", (PyCFunction) rnd_get_state, METH_O, NULL },
    { "rnd_seed", (PyCFunction) rnd_seed, METH_VARARGS, NULL },
    { "rnd_set_state", (PyCFunction) rnd_set_state, METH_VARARGS, NULL },
    { "rnd_shuffle", (PyCFunction) rnd_shuffle, METH_O, NULL },
    { NULL },
};


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

    if (_rnd_random_seed(&py_random_state) ||
        _rnd_random_seed(&np_random_state))
        return MOD_ERROR_VAL;

    return MOD_SUCCESS_VAL(m);
}
