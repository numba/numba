#include "_pymodule.h"
#include "longintrepr.h"

#include "pymacro.h"
#include "pyctype.c"

/* from https://github.com/python/cpython/blob/2fb2bc81c3f40d73945c6102569495140e1182c7/Objects/longobject.c#L125 */
static PyLongObject *
long_normalize(PyLongObject *v)
{
    Py_ssize_t j = Py_ABS(Py_SIZE(v));
    Py_ssize_t i = j;

    while (i > 0 && v->ob_digit[i-1] == 0)
        --i;
    if (i != j)
        Py_SIZE(v) = (Py_SIZE(v) < 0) ? -(i) : i;
    return v;
}

#ifndef NSMALLPOSINTS
#define NSMALLPOSINTS           257
#endif
#ifndef NSMALLNEGINTS
#define NSMALLNEGINTS           5
#endif

/* convert a PyLong of size 1, 0 or -1 to an sdigit */
#define MEDIUM_VALUE(x) (assert(-1 <= Py_SIZE(x) && Py_SIZE(x) <= 1),   \
         Py_SIZE(x) < 0 ? -(sdigit)(x)->ob_digit[0] :   \
             (Py_SIZE(x) == 0 ? (sdigit)0 :                             \
              (sdigit)(x)->ob_digit[0]))

#define MAX_LONG_DIGITS \
    ((PY_SSIZE_T_MAX - offsetof(PyLongObject, ob_digit))/sizeof(digit))

static PyLongObject small_ints[NSMALLNEGINTS + NSMALLPOSINTS];

/* from https://github.com/python/cpython/blob/2fb2bc81c3f40d73945c6102569495140e1182c7/Objects/longobject.c#L48 */
static PyObject *
get_small_int(sdigit ival)
{
    PyObject *v;
    assert(-NSMALLNEGINTS <= ival && ival < NSMALLPOSINTS);
    v = (PyObject *)&small_ints[ival + NSMALLNEGINTS];
    Py_INCREF(v);
#ifdef COUNT_ALLOCS
    if (ival >= 0)
        _Py_quick_int_allocs++;
    else
        _Py_quick_neg_int_allocs++;
#endif
    return v;
}

/* from https://github.com/python/cpython/blob/2fb2bc81c3f40d73945c6102569495140e1182c7/Objects/longobject.c#L69 */
static PyLongObject *
maybe_small_long(PyLongObject *v)
{
    if (v && Py_ABS(Py_SIZE(v)) <= 1) {
        sdigit ival = MEDIUM_VALUE(v);
        if (-NSMALLNEGINTS <= ival && ival < NSMALLPOSINTS) {
            Py_DECREF(v);
            return (PyLongObject *)get_small_int(ival);
        }
    }
    return v;
}

/* from https://github.com/python/cpython/blob/2fb2bc81c3f40d73945c6102569495140e1182c7/Objects/longobject.c#L2184 */
static int
long_from_binary_base(const char **str, int base, PyLongObject **res)
{
    const char *p = *str;
    const char *start = p;
    char prev = 0;
    Py_ssize_t digits = 0;
    int bits_per_char;
    Py_ssize_t n;
    PyLongObject *z;
    twodigits accum;
    int bits_in_accum;
    digit *pdigit;

    assert(base >= 2 && base <= 32 && (base & (base - 1)) == 0);
    n = base;
    for (bits_per_char = -1; n; ++bits_per_char) {
        n >>= 1;
    }
    /* count digits and set p to end-of-string */
    while (_PyLong_DigitValue[Py_CHARMASK(*p)] < base || *p == '_') {
        if (*p == '_') {
            if (prev == '_') {
                *str = p - 1;
                return -1;
            }
        } else {
            ++digits;
        }
        prev = *p;
        ++p;
    }
    if (prev == '_') {
        /* Trailing underscore not allowed. */
        *str = p - 1;
        return -1;
    }

    *str = p;
    /* n <- the number of Python digits needed,
            = ceiling((digits * bits_per_char) / PyLong_SHIFT). */
    if (digits > (PY_SSIZE_T_MAX - (PyLong_SHIFT - 1)) / bits_per_char) {
        PyErr_SetString(PyExc_ValueError,
                        "int string too large to convert");
        *res = NULL;
        return 0;
    }
    n = (digits * bits_per_char + PyLong_SHIFT - 1) / PyLong_SHIFT;
    z = _PyLong_New(n);
    if (z == NULL) {
        *res = NULL;
        return 0;
    }
    /* Read string from right, and fill in int from left; i.e.,
     * from least to most significant in both.
     */
    accum = 0;
    bits_in_accum = 0;
    pdigit = z->ob_digit;
    while (--p >= start) {
        int k;
        if (*p == '_') {
            continue;
        }
        k = (int)_PyLong_DigitValue[Py_CHARMASK(*p)];
        assert(k >= 0 && k < base);
        accum |= (twodigits)k << bits_in_accum;
        bits_in_accum += bits_per_char;
        if (bits_in_accum >= PyLong_SHIFT) {
            *pdigit++ = (digit)(accum & PyLong_MASK);
            assert(pdigit - z->ob_digit <= n);
            accum >>= PyLong_SHIFT;
            bits_in_accum -= PyLong_SHIFT;
            assert(bits_in_accum < PyLong_SHIFT);
        }
    }
    if (bits_in_accum) {
        assert(bits_in_accum <= PyLong_SHIFT);
        *pdigit++ = (digit)accum;
        assert(pdigit - z->ob_digit <= n);
    }
    while (pdigit - z->ob_digit < n)
        *pdigit++ = 0;
    *res = long_normalize(z);
    return 0;
}


/* from https://github.com/python/cpython/blob/2fb2bc81c3f40d73945c6102569495140e1182c7/Objects/longobject.c#L2279 */
/* Parses an int from a bytestring. Leading and trailing whitespace will be
 * ignored.
 *
 * If successful, a PyLong object will be returned and 'pend' will be pointing
 * to the first unused byte unless it's NULL.
 *
 * If unsuccessful, NULL will be returned.
 */
PyObject *
PyLong_FromString_ext(const char *str, char **pend, int base)
{
    int sign = 1, error_if_nonzero = 0;
    const char *start, *orig_str = str;
    PyLongObject *z = NULL;
    PyObject *strobj;
    Py_ssize_t slen;

    if ((base != 0 && base < 2) || base > 36) {
        PyErr_SetString(PyExc_ValueError,
                        "int() arg 2 must be >= 2 and <= 36");
        return NULL;
    }
    while (*str != '\0' && Py_ISSPACE(Py_CHARMASK(*str))) {
        str++;
    }
    if (*str == '+') {
        ++str;
    }
    else if (*str == '-') {
        ++str;
        sign = -1;
    }
    if (base == 0) {
        if (str[0] != '0') {
            base = 10;
        }
        else if (str[1] == 'x' || str[1] == 'X') {
            base = 16;
        }
        else if (str[1] == 'o' || str[1] == 'O') {
            base = 8;
        }
        else if (str[1] == 'b' || str[1] == 'B') {
            base = 2;
        }
        else {
            /* "old" (C-style) octal literal, now invalid.
               it might still be zero though */
            error_if_nonzero = 1;
            base = 10;
        }
    }
    if (str[0] == '0' &&
        ((base == 16 && (str[1] == 'x' || str[1] == 'X')) ||
         (base == 8  && (str[1] == 'o' || str[1] == 'O')) ||
         (base == 2  && (str[1] == 'b' || str[1] == 'B')))) {
        str += 2;
        /* One underscore allowed here. */
        if (*str == '_') {
            ++str;
        }
    }
    if (str[0] == '_') {
        /* May not start with underscores. */
        goto onError;
    }

    start = str;
    if ((base & (base - 1)) == 0) {
        int res = long_from_binary_base(&str, base, &z);
        if (res < 0) {
            /* Syntax error. */
            goto onError;
        }
    }
    else {
/***
Binary bases can be converted in time linear in the number of digits, because
Python's representation base is binary.  Other bases (including decimal!) use
the simple quadratic-time algorithm below, complicated by some speed tricks.

First some math:  the largest integer that can be expressed in N base-B digits
is B**N-1.  Consequently, if we have an N-digit input in base B, the worst-
case number of Python digits needed to hold it is the smallest integer n s.t.

    BASE**n-1 >= B**N-1  [or, adding 1 to both sides]
    BASE**n >= B**N      [taking logs to base BASE]
    n >= log(B**N)/log(BASE) = N * log(B)/log(BASE)

The static array log_base_BASE[base] == log(base)/log(BASE) so we can compute
this quickly.  A Python int with that much space is reserved near the start,
and the result is computed into it.

The input string is actually treated as being in base base**i (i.e., i digits
are processed at a time), where two more static arrays hold:

    convwidth_base[base] = the largest integer i such that base**i <= BASE
    convmultmax_base[base] = base ** convwidth_base[base]

The first of these is the largest i such that i consecutive input digits
must fit in a single Python digit.  The second is effectively the input
base we're really using.

Viewing the input as a sequence <c0, c1, ..., c_n-1> of digits in base
convmultmax_base[base], the result is "simply"

   (((c0*B + c1)*B + c2)*B + c3)*B + ... ))) + c_n-1

where B = convmultmax_base[base].

Error analysis:  as above, the number of Python digits `n` needed is worst-
case

    n >= N * log(B)/log(BASE)

where `N` is the number of input digits in base `B`.  This is computed via

    size_z = (Py_ssize_t)((scan - str) * log_base_BASE[base]) + 1;

below.  Two numeric concerns are how much space this can waste, and whether
the computed result can be too small.  To be concrete, assume BASE = 2**15,
which is the default (and it's unlikely anyone changes that).

Waste isn't a problem:  provided the first input digit isn't 0, the difference
between the worst-case input with N digits and the smallest input with N
digits is about a factor of B, but B is small compared to BASE so at most
one allocated Python digit can remain unused on that count.  If
N*log(B)/log(BASE) is mathematically an exact integer, then truncating that
and adding 1 returns a result 1 larger than necessary.  However, that can't
happen:  whenever B is a power of 2, long_from_binary_base() is called
instead, and it's impossible for B**i to be an integer power of 2**15 when
B is not a power of 2 (i.e., it's impossible for N*log(B)/log(BASE) to be
an exact integer when B is not a power of 2, since B**i has a prime factor
other than 2 in that case, but (2**15)**j's only prime factor is 2).

The computed result can be too small if the true value of N*log(B)/log(BASE)
is a little bit larger than an exact integer, but due to roundoff errors (in
computing log(B), log(BASE), their quotient, and/or multiplying that by N)
yields a numeric result a little less than that integer.  Unfortunately, "how
close can a transcendental function get to an integer over some range?"
questions are generally theoretically intractable.  Computer analysis via
continued fractions is practical:  expand log(B)/log(BASE) via continued
fractions, giving a sequence i/j of "the best" rational approximations.  Then
j*log(B)/log(BASE) is approximately equal to (the integer) i.  This shows that
we can get very close to being in trouble, but very rarely.  For example,
76573 is a denominator in one of the continued-fraction approximations to
log(10)/log(2**15), and indeed:

    >>> log(10)/log(2**15)*76573
    16958.000000654003

is very close to an integer.  If we were working with IEEE single-precision,
rounding errors could kill us.  Finding worst cases in IEEE double-precision
requires better-than-double-precision log() functions, and Tim didn't bother.
Instead the code checks to see whether the allocated space is enough as each
new Python digit is added, and copies the whole thing to a larger int if not.
This should happen extremely rarely, and in fact I don't have a test case
that triggers it(!).  Instead the code was tested by artificially allocating
just 1 digit at the start, so that the copying code was exercised for every
digit beyond the first.
***/
        twodigits c;           /* current input character */
        Py_ssize_t size_z;
        Py_ssize_t digits = 0;
        int i;
        int convwidth;
        twodigits convmultmax, convmult;
        digit *pz, *pzstop;
        const char *scan, *lastdigit;
        char prev = 0;

        static double log_base_BASE[37] = {0.0e0,};
        static int convwidth_base[37] = {0,};
        static twodigits convmultmax_base[37] = {0,};
        double fsize_z;

        if (log_base_BASE[base] == 0.0) {
            twodigits convmax = base;
            int i = 1;

            log_base_BASE[base] = (log((double)base) /
                                   log((double)PyLong_BASE));
            for (;;) {
                twodigits next = convmax * base;
                if (next > PyLong_BASE) {
                    break;
                }
                convmax = next;
                ++i;
            }
            convmultmax_base[base] = convmax;
            assert(i > 0);
            convwidth_base[base] = i;
        }

        /* Find length of the string of numeric characters. */
        scan = str;
        lastdigit = str;

        while (_PyLong_DigitValue[Py_CHARMASK(*scan)] < base || *scan == '_') {
            if (*scan == '_') {
                if (prev == '_') {
                    /* Only one underscore allowed. */
                    str = lastdigit + 1;
                    goto onError;
                }
            }
            else {
                ++digits;
                lastdigit = scan;
            }
            prev = *scan;
            ++scan;
        }
        if (prev == '_') {
            /* Trailing underscore not allowed. */
            /* Set error pointer to first underscore. */
            str = lastdigit + 1;
            goto onError;
        }

        /* Create an int object that can contain the largest possible
         * integer with this base and length.  Note that there's no
         * need to initialize z->ob_digit -- no slot is read up before
         * being stored into.
         */
        fsize_z = (double)digits * log_base_BASE[base] + 1.0;
        if (fsize_z > (double)MAX_LONG_DIGITS) {
            /* The same exception as in _PyLong_New(). */
            PyErr_SetString(PyExc_OverflowError,
                            "too many digits in integer");
            return NULL;
        }
        size_z = (Py_ssize_t)fsize_z;
        /* Uncomment next line to test exceedingly rare copy code */
        /* size_z = 1; */
        assert(size_z > 0);
        z = _PyLong_New(size_z);
        if (z == NULL) {
            return NULL;
        }
        Py_SIZE(z) = 0;

        /* `convwidth` consecutive input digits are treated as a single
         * digit in base `convmultmax`.
         */
        convwidth = convwidth_base[base];
        convmultmax = convmultmax_base[base];

        /* Work ;-) */
        while (str < scan) {
            if (*str == '_') {
                str++;
                continue;
            }
            /* grab up to convwidth digits from the input string */
            c = (digit)_PyLong_DigitValue[Py_CHARMASK(*str++)];
            for (i = 1; i < convwidth && str != scan; ++str) {
                if (*str == '_') {
                    continue;
                }
                i++;
                c = (twodigits)(c *  base +
                                (int)_PyLong_DigitValue[Py_CHARMASK(*str)]);
                assert(c < PyLong_BASE);
            }

            convmult = convmultmax;
            /* Calculate the shift only if we couldn't get
             * convwidth digits.
             */
            if (i != convwidth) {
                convmult = base;
                for ( ; i > 1; --i) {
                    convmult *= base;
                }
            }

            /* Multiply z by convmult, and add c. */
            pz = z->ob_digit;
            pzstop = pz + Py_SIZE(z);
            for (; pz < pzstop; ++pz) {
                c += (twodigits)*pz * convmult;
                *pz = (digit)(c & PyLong_MASK);
                c >>= PyLong_SHIFT;
            }
            /* carry off the current end? */
            if (c) {
                assert(c < PyLong_BASE);
                if (Py_SIZE(z) < size_z) {
                    *pz = (digit)c;
                    ++Py_SIZE(z);
                }
                else {
                    PyLongObject *tmp;
                    /* Extremely rare.  Get more space. */
                    assert(Py_SIZE(z) == size_z);
                    tmp = _PyLong_New(size_z + 1);
                    if (tmp == NULL) {
                        Py_DECREF(z);
                        return NULL;
                    }
                    memcpy(tmp->ob_digit,
                           z->ob_digit,
                           sizeof(digit) * size_z);
                    Py_DECREF(z);
                    z = tmp;
                    z->ob_digit[size_z] = (digit)c;
                    ++size_z;
                }
            }
        }
    }
        
    if (z == NULL) {
        return NULL;
    }
    if (error_if_nonzero) {
        /* reset the base to 0, else the exception message
           doesn't make too much sense */
        base = 0;
        if (Py_SIZE(z) != 0) {
            goto onError;
        }
        /* there might still be other problems, therefore base
           remains zero here for the same reason */
    }
    if (str == start) {
        goto onError;
    }
    if (sign < 0) {
        Py_SIZE(z) = -(Py_SIZE(z));
    }
        
    while (*str && Py_ISSPACE(Py_CHARMASK(*str))) {
        str++;
    }
    if (*str != '\0') {
        goto onError;
    }
    long_normalize(z);
        
    //!!!z = maybe_small_long(z);
    if (z == NULL) {
        return NULL;
    }
    if (pend != NULL) {
        *pend = (char *)str;
    }
    return (PyObject *) z;

  onError:
    if (pend != NULL) {
        *pend = (char *)str;
    }
    Py_XDECREF(z);
    slen = strlen(orig_str) < 200 ? strlen(orig_str) : 200;
    strobj = PyUnicode_FromStringAndSize(orig_str, slen);
    if (strobj == NULL) {
        return NULL;
    }
    PyErr_Format(PyExc_ValueError,
                 "invalid literal for int() with base %d: %.200R",
                 base, strobj);
    Py_DECREF(strobj);
    return NULL;
}

/* from https://github.com/python/cpython/blob/2fb2bc81c3f40d73945c6102569495140e1182c7/Objects/longobject.c#L2640 */
/* Since PyLong_FromString doesn't have a length parameter,
 * check here for possible NULs in the string.
 *
 * Reports an invalid literal as a bytes object.
 */
PyObject *
_PyLong_FromBytes_ext(const char *s, Py_ssize_t len, int base)
{
    PyObject *result, *strobj;
    char *end = NULL;

    result = PyLong_FromString_ext(s, &end, base);
    if (end == NULL || (result != NULL && end == s + len))
        return result;
    Py_XDECREF(result);
    strobj = PyBytes_FromStringAndSize(s, Py_MIN(len, 200));
    if (strobj != NULL) {
        PyErr_Format(PyExc_ValueError,
                     "invalid literal for int() with base %d: %.200R",
                     base, strobj);
        Py_DECREF(strobj);
    }
    return NULL;
}

long str2int(char* str, long len, int base)
{
    return PyLong_AsLong(_PyLong_FromBytes_ext(str, len, base));
}

MOD_INIT(string_conversion_ext) {
    PyObject *m;
    MOD_DEF(m, "int_from_string_ext", "No docs", NULL)
        if (m == NULL)
            return MOD_ERROR_VAL;
    PyModule_AddObject(m, "str2int", PyLong_FromVoidPtr(&str2int));
    return MOD_SUCCESS_VAL(m);
}
