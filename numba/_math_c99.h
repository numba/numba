#ifndef NUMBA_MATH_C99_H_
#define NUMBA_MATH_C99_H_

#include "_numba_common.h"

VISIBILITY_HIDDEN double m_acosh(double x);
VISIBILITY_HIDDEN float m_acoshf(float x);

VISIBILITY_HIDDEN double m_asinh(double x);
VISIBILITY_HIDDEN float m_asinhf(float x);

VISIBILITY_HIDDEN double m_atanh(double x);
VISIBILITY_HIDDEN float m_atanhf(float x);

VISIBILITY_HIDDEN double m_expm1(double x);
VISIBILITY_HIDDEN float m_expm1f(float x);

VISIBILITY_HIDDEN double m_log1p(double x);
VISIBILITY_HIDDEN float m_log1pf(float x);

VISIBILITY_HIDDEN double m_round(double x);
VISIBILITY_HIDDEN float m_roundf(float x);

VISIBILITY_HIDDEN double m_trunc(double x);
VISIBILITY_HIDDEN float m_truncf(float x);

VISIBILITY_HIDDEN double m_atan2(double y, double x);
VISIBILITY_HIDDEN float m_atan2f(float y, float x);

#ifdef _MSC_VER

/* Define missing (C99) math functions for Windows */

#define asinh(x) m_asinh(x)
#define asinhf(x) m_asinhf(x)
#define acosh(x) m_acosh(x)
#define acoshf(x) m_acoshf(x)
#define atanh(x) m_atanh(x)
#define atanhf(x) m_atanhf(x)

#define expm1(x) m_expm1(x)
#define expm1f(x) m_expm1f(x)
#define log1p(x) m_log1p(x)
#define log1pf(x) m_log1pf(x)

#define round(x) m_round(x)
#define roundf(x) m_roundf(x)
#define trunc(x) m_trunc(x)
#define truncf(x) m_truncf(x)

#define atan2f(x, y) m_atan2f(x, y)

#endif /* _MSC_VER */

#define atan2_fixed(x, y) m_atan2(x, y)

#endif /* NUMBA_MATH_C99_H_ */
