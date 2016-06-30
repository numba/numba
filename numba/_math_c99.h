#ifndef NUMBA_MATH_C99_H_
#define NUMBA_MATH_C99_H_

#include "_numba_common.h"

/* We require C99 on POSIX, but have to be tolerant on Windows since
   Python < 3.5 is compiled with old MSVC versions */

#if !defined(_MSC_VER) || _MSC_VER >= 1800  /* Visual Studio 2013 */
#define HAVE_C99_MATH 1
#else
#define HAVE_C99_MATH 0
#endif


VISIBILITY_HIDDEN double m_acosh(double x);
VISIBILITY_HIDDEN float m_acoshf(float x);

VISIBILITY_HIDDEN double m_asinh(double x);
VISIBILITY_HIDDEN float m_asinhf(float x);

VISIBILITY_HIDDEN double m_atanh(double x);
VISIBILITY_HIDDEN float m_atanhf(float x);

VISIBILITY_HIDDEN double m_erf(double x);
VISIBILITY_HIDDEN float m_erff(float x);

VISIBILITY_HIDDEN double m_erfc(double x);
VISIBILITY_HIDDEN float m_erfcf(float x);

VISIBILITY_HIDDEN double m_expm1(double x);
VISIBILITY_HIDDEN float m_expm1f(float x);

VISIBILITY_HIDDEN double m_gamma(double x);
VISIBILITY_HIDDEN float m_gammaf(float x);

VISIBILITY_HIDDEN double m_lgamma(double x);
VISIBILITY_HIDDEN float m_lgammaf(float x);

VISIBILITY_HIDDEN double m_log1p(double x);
VISIBILITY_HIDDEN float m_log1pf(float x);

VISIBILITY_HIDDEN double m_round(double x);
VISIBILITY_HIDDEN float m_roundf(float x);

VISIBILITY_HIDDEN double m_trunc(double x);
VISIBILITY_HIDDEN float m_truncf(float x);

VISIBILITY_HIDDEN double m_atan2(double y, double x);
VISIBILITY_HIDDEN float m_atan2f(float y, float x);


#if !HAVE_C99_MATH

/* Define missing math functions */

#define asinh(x) m_asinh(x)
#define asinhf(x) m_asinhf(x)
#define acosh(x) m_acosh(x)
#define acoshf(x) m_acoshf(x)
#define atanh(x) m_atanh(x)
#define atanhf(x) m_atanhf(x)

#define erf(x) m_erf(x)
#define erfc(x) m_erfc(x)
#define erfcf(x) m_erfcf(x)
#define erff(x) m_erff(x)

#define expm1(x) m_expm1(x)
#define expm1f(x) m_expm1f(x)
#define log1p(x) m_log1p(x)
#define log1pf(x) m_log1pf(x)

#define lgamma(x) m_lgamma(x)
#define lgammaf(x) m_lgammaf(x)
#define tgamma(x) m_gamma(x)
#define tgammaf(x) m_gammaf(x)

#define round(x) m_round(x)
#define roundf(x) m_roundf(x)
#define trunc(x) m_trunc(x)
#define truncf(x) m_truncf(x)

#define atan2f(x, y) m_atan2f(x, y)

#endif /* !HAVE_C99_MATH */

#define atan2_fixed(x, y) m_atan2(x, y)

#endif /* NUMBA_MATH_C99_H_ */
