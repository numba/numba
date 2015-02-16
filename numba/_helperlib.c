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

/* For Numpy 1.6 */
#ifndef NPY_ARRAY_BEHAVED
    #define NPY_ARRAY_BEHAVED NPY_BEHAVED
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

/*
 * Fill in the *arystruct* with information from the Numpy array *obj*.
 * *arystruct*'s layout is defined in numba.targets.arrayobj (look
 * for the ArrayTemplate class).
 */

typedef struct {
    PyObject *parent;
    npy_intp nitems;
    npy_intp itemsize;
    void *data;
    npy_intp shape_and_strides[];
} arystruct_t;

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

    return 0;
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
void Numba_release_record_buffer(Py_buffer *buf)
{
    PyBuffer_Release(buf);
}


static
void Numba_gil_ensure(PyGILState_STATE *state) {
    *state = PyGILState_Ensure();
}

static
void Numba_gil_release(PyGILState_STATE *state) {
    PyGILState_Release(*state);
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

#define declmethod(func) do {                          \
    PyObject *val = PyLong_FromVoidPtr(&Numba_##func); \
    if (val == NULL) goto error;                       \
    if (PyDict_SetItemString(dct, #func, val)) {       \
        Py_DECREF(val);                                \
        goto error;                                    \
    }                                                  \
    Py_DECREF(val);                                    \
} while (0)

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
    declmethod(extract_record_data);
    declmethod(release_record_buffer);
    declmethod(adapt_ndarray);
    declmethod(ndarray_new);
    declmethod(extract_np_datetime);
    declmethod(create_np_datetime);
    declmethod(extract_np_timedelta);
    declmethod(create_np_timedelta);
    declmethod(recreate_record);
    declmethod(fptoui);
    declmethod(fptouif);
    declmethod(gil_ensure);
    declmethod(gil_release);
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

    return MOD_SUCCESS_VAL(m);
}
