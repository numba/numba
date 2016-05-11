#include "Python.h"
#include <float.h>
#include "_math_c99.h"


/* Copied from Python Module/_math.c with modification to symbol name */


/* The following copyright notice applies to the original
   implementations of acosh, asinh and atanh. */

/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */

static const double ln2 = 6.93147180559945286227E-01;
static const double two_pow_m28 = 3.7252902984619141E-09; /* 2**-28 */
static const double two_pow_p28 = 268435456.0; /* 2**28 */
static const double zero = 0.0;

/* acosh(x)
 * Method :
 *      Based on
 *            acosh(x) = log [ x + sqrt(x*x-1) ]
 *      we have
 *            acosh(x) := log(x)+ln2, if x is large; else
 *            acosh(x) := log(2x-1/(sqrt(x*x-1)+x)) if x>2; else
 *            acosh(x) := log1p(t+sqrt(2.0*t+t*t)); where t=x-1.
 *
 * Special cases:
 *      acosh(x) is NaN with signal if x<1.
 *      acosh(NaN) is NaN without signal.
 */

double
m_acosh(double x)
{
    if (Py_IS_NAN(x)) {
        return x+x;
    }
    if (x < 1.) {                       /* x < 1;  return a signaling NaN */
        errno = EDOM;
#ifdef Py_NAN
        return Py_NAN;
#else
        return (x-x)/(x-x);
#endif
    }
    else if (x >= two_pow_p28) {        /* x > 2**28 */
        if (Py_IS_INFINITY(x)) {
            return x+x;
        }
        else {
            return log(x)+ln2;          /* acosh(huge)=log(2x) */
        }
    }
    else if (x == 1.) {
        return 0.0;                     /* acosh(1) = 0 */
    }
    else if (x > 2.) {                  /* 2 < x < 2**28 */
        double t = x*x;
        return log(2.0*x - 1.0 / (x + sqrt(t - 1.0)));
    }
    else {                              /* 1 < x <= 2 */
        double t = x - 1.0;
        return m_log1p(t + sqrt(2.0*t + t*t));
    }
}


/* asinh(x)
 * Method :
 *      Based on
 *              asinh(x) = sign(x) * log [ |x| + sqrt(x*x+1) ]
 *      we have
 *      asinh(x) := x  if  1+x*x=1,
 *               := sign(x)*(log(x)+ln2)) for large |x|, else
 *               := sign(x)*log(2|x|+1/(|x|+sqrt(x*x+1))) if|x|>2, else
 *               := sign(x)*log1p(|x| + x^2/(1 + sqrt(1+x^2)))
 */

double
m_asinh(double x)
{
    double w;
    double absx = fabs(x);

    if (Py_IS_NAN(x) || Py_IS_INFINITY(x)) {
        return x+x;
    }
    if (absx < two_pow_m28) {           /* |x| < 2**-28 */
        return x;                       /* return x inexact except 0 */
    }
    if (absx > two_pow_p28) {           /* |x| > 2**28 */
        w = log(absx)+ln2;
    }
    else if (absx > 2.0) {              /* 2 < |x| < 2**28 */
        w = log(2.0*absx + 1.0 / (sqrt(x*x + 1.0) + absx));
    }
    else {                              /* 2**-28 <= |x| < 2= */
        double t = x*x;
        w = m_log1p(absx + t / (1.0 + sqrt(1.0 + t)));
    }
    return copysign(w, x);

}

/* atanh(x)
 * Method :
 *    1.Reduced x to positive by atanh(-x) = -atanh(x)
 *    2.For x>=0.5
 *                  1              2x                          x
 *      atanh(x) = --- * log(1 + -------) = 0.5 * log1p(2 * -------)
 *                  2             1 - x                      1 - x
 *
 *      For x<0.5
 *      atanh(x) = 0.5*log1p(2x+2x*x/(1-x))
 *
 * Special cases:
 *      atanh(x) is NaN if |x| >= 1 with signal;
 *      atanh(NaN) is that NaN with no signal;
 *
 */

double
m_atanh(double x)
{
    double absx;
    double t;

    if (Py_IS_NAN(x)) {
        return x+x;
    }
    absx = fabs(x);
    if (absx >= 1.) {                   /* |x| >= 1 */
        errno = EDOM;
#ifdef Py_NAN
        return Py_NAN;
#else
        return x/zero;
#endif
    }
    if (absx < two_pow_m28) {           /* |x| < 2**-28 */
        return x;
    }
    if (absx < 0.5) {                   /* |x| < 0.5 */
        t = absx+absx;
        t = 0.5 * m_log1p(t + t*absx / (1.0 - absx));
    }
    else {                              /* 0.5 <= |x| <= 1.0 */
        t = 0.5 * m_log1p((absx + absx) / (1.0 - absx));
    }
    return copysign(t, x);
}

/* Mathematically, expm1(x) = exp(x) - 1.  The expm1 function is designed
   to avoid the significant loss of precision that arises from direct
   evaluation of the expression exp(x) - 1, for x near 0. */

double
m_expm1(double x)
{
    /* For abs(x) >= log(2), it's safe to evaluate exp(x) - 1 directly; this
       also works fine for infinities and nans.

       For smaller x, we can use a method due to Kahan that achieves close to
       full accuracy.
    */

    if (fabs(x) < 0.7) {
        double u;
        u = exp(x);
        if (u == 1.0)
            return x;
        else
            return (u - 1.0) * x / log(u);
    }
    else
        return exp(x) - 1.0;
}

/* log1p(x) = log(1+x).  The log1p function is designed to avoid the
   significant loss of precision that arises from direct evaluation when x is
   small. */

double
m_log1p(double x)
{
    /* For x small, we use the following approach.  Let y be the nearest float
       to 1+x, then

         1+x = y * (1 - (y-1-x)/y)

       so log(1+x) = log(y) + log(1-(y-1-x)/y).  Since (y-1-x)/y is tiny, the
       second term is well approximated by (y-1-x)/y.  If abs(x) >=
       DBL_EPSILON/2 or the rounding-mode is some form of round-to-nearest
       then y-1-x will be exactly representable, and is computed exactly by
       (y-1)-x.

       If abs(x) < DBL_EPSILON/2 and the rounding mode is not known to be
       round-to-nearest then this method is slightly dangerous: 1+x could be
       rounded up to 1+DBL_EPSILON instead of down to 1, and in that case
       y-1-x will not be exactly representable any more and the result can be
       off by many ulps.  But this is easily fixed: for a floating-point
       number |x| < DBL_EPSILON/2., the closest floating-point number to
       log(1+x) is exactly x.
    */

    double y;
    if (fabs(x) < DBL_EPSILON/2.) {
        return x;
    }
    else if (-0.5 <= x && x <= 1.) {
        /* WARNING: it's possible than an overeager compiler
           will incorrectly optimize the following two lines
           to the equivalent of "return log(1.+x)". If this
           happens, then results from log1p will be inaccurate
           for small x. */
        y = 1.+x;
        return log(y)-((y-1.)-x)/y;
    }
    else {
        /* NaNs and infinities should end up here */
        return log(1.+x);
    }
}

/* Hand written */
double m_trunc(double x)
{
    double integral;
    (void)modf(x, &integral);
    return integral;
}


/* Hand written */
double m_round(double x) {
    if (x < 0.0) {
        return ceil(x - 0.5);
    } else {
        return floor(x + 0.5);
    }
}

/* Hand written */
float m_roundf(float x) {
    if (x < 0.0) {
        return (float) ceilf(x - 0.5f);
    } else {
        return (float) floorf(x + 0.5f);
    }
}

/*
   CPython implementation for atan2():

   wrapper for atan2 that deals directly with special cases before
   delegating to the platform libm for the remaining cases.  This
   is necessary to get consistent behaviour across platforms.
   Windows, FreeBSD and alpha Tru64 are amongst platforms that don't
   always follow C99.
*/

double m_atan2(double y, double x)
{
    if (Py_IS_NAN(x) || Py_IS_NAN(y))
        return Py_NAN;
    if (Py_IS_INFINITY(y)) {
        if (Py_IS_INFINITY(x)) {
            if (copysign(1., x) == 1.)
                /* atan2(+-inf, +inf) == +-pi/4 */
                return copysign(0.25*Py_MATH_PI, y);
            else
                /* atan2(+-inf, -inf) == +-pi*3/4 */
                return copysign(0.75*Py_MATH_PI, y);
        }
        /* atan2(+-inf, x) == +-pi/2 for finite x */
        return copysign(0.5*Py_MATH_PI, y);
    }
    if (Py_IS_INFINITY(x) || y == 0.) {
        if (copysign(1., x) == 1.)
            /* atan2(+-y, +inf) = atan2(+-0, +x) = +-0. */
            return copysign(0., y);
        else
            /* atan2(+-y, -inf) = atan2(+-0., -x) = +-pi. */
            return copysign(Py_MATH_PI, y);
    }
    return atan2(y, x);
}

/* Map to double version directly */
float m_atan2f(float y, float x) {
    return (float) m_atan2(y, x);
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

double
m_gamma(double x)
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

/*
   lgamma:  natural log of the absolute value of the Gamma function.
   For large arguments, Lanczos' formula works extremely well here.
*/

double
m_lgamma(double x)
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

double
m_erf(double x)
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

/* Complementary error function erfc(x), for general x. */

double
m_erfc(double x)
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

#define FLOATVER(Fn) float Fn##f(float x) { return (float)Fn(x); }

FLOATVER(m_acosh);
FLOATVER(m_asinh);
FLOATVER(m_atanh);
FLOATVER(m_erf);
FLOATVER(m_erfc);
FLOATVER(m_expm1);
FLOATVER(m_gamma);
FLOATVER(m_lgamma);
FLOATVER(m_log1p);
FLOATVER(m_trunc);

