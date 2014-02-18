#ifndef NUMBA_MATH_C99_H_
#define NUMBA_MATH_C99_H_

double m_acosh(double x);
float m_acoshf(float x);

double m_asinh(double x);
float m_asinhf(float x);

double m_atanh(double x);
float m_atanhf(float x);

double m_expm1(double x);
float m_expm1f(float x);

double m_log1p(double x);
float m_log1pf(float x);

double m_round(double x);
float m_roundf(float x);

double m_trunc(double x);
float m_truncf(float x);

float m_atan2f(float y, float x);

#ifdef _MSC_VER

/* define asinh acosh atanh for windows */

#define asinh m_asinh
#define asinhf m_asinhf
#define acosh m_acosh
#define acoshf m_acoshf
#define atanh m_atanh
#define atanhf m_atanhf


#define expm1 m_expm1
#define expm1f m_expm1f
#define log1p m_log1p
#define log1pf m_log1pf

#define round m_round
#define roundf m_roundf
#define trunc m_trunc
#define truncf m_truncf

#define atan2f m_atan2f
/* provide floating point equivalence */

#endif /* _MSC_VER */

#endif /* NUMBA_MATH_C99_H_ */