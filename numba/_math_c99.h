#ifndef NUMBA_MATH_C99_H_
#define NUMBA_MATH_C99_H_

double m_acosh(double x);
double m_asinh(double x);
double m_atanh(double x);
double m_expm1(double x);
double m_log1p(double x);

#ifdef _MSC_VER

/* define asinh acosh atanh for windows */

#define asinh m_asinh
#define acosh m_acosh
#define atanh m_atanh

/* define expm1 log1p */

#define expm1 m_expm1
#define log1p m_log1p

/* provide floating point equivalence */

float asinhf(float x) {
    return asinh(x);
}

float acoshf(float x) {
    return acosh(x);
}

float atanhf(float x) {
    return atanh(x);
}

float expm1f(float x) {
    return expm1(x);
}

float log1pf(float x) {
    return log1p(x);
}

#endif /* _MSC_VER */

#endif /* NUMBA_MATH_C99_H_ */