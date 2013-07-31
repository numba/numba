/* Utilities copied from Cython */

#ifndef NUMBA_INLINE
  #if defined(__GNUC__)
    #define NUMBA_INLINE __inline__
  #elif defined(_MSC_VER)
    #define NUMBA_INLINE __inline
  #elif defined (__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
    #define NUMBA_INLINE inline
  #else
    #define NUMBA_INLINE
  #endif
#endif

#ifndef NUMBA_UNUSED
# if defined(__GNUC__)
#   if !(defined(__cplusplus)) || (__GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 4))
#     define NUMBA_UNUSED __attribute__ ((__unused__))
#   else
#     define NUMBA_UNUSED
#   endif
# elif defined(__ICC) || (defined(__INTEL_COMPILER) && !defined(_MSC_VER))
#   define NUMBA_UNUSED __attribute__ ((__unused__))
# else
#   define NUMBA_UNUSED
# endif
#endif

#if PY_VERSION_HEX < 0x02050000
    #define CYTHON_FORMAT_SSIZE_T ""
    #define __Numba_NAMESTR(n) ((char *)(n))
    #define __Numba_DOCSTR(n)  ((char *)(n))
#else
    #define CYTHON_FORMAT_SSIZE_T "z"
    #define __Numba_NAMESTR(n) (n)
    #define __Numba_DOCSTR(n)  (n)
#endif

#if PY_VERSION_HEX > 0x03030000 && defined(PyUnicode_KIND)
  #define CYTHON_PEP393_ENABLED 1
  #define __Numba_PyUnicode_READY(op)       (likely(PyUnicode_IS_READY(op)) ? \
                                              0 : _PyUnicode_Ready((PyObject *)(op)))
  #define __Numba_PyUnicode_GET_LENGTH(u)   PyUnicode_GET_LENGTH(u)
  #define __Numba_PyUnicode_READ_CHAR(u, i) PyUnicode_READ_CHAR(u, i)
  #define __Numba_PyUnicode_READ(k, d, i)   PyUnicode_READ(k, d, i)
#else
  #define CYTHON_PEP393_ENABLED 0
  #define __Numba_PyUnicode_READY(op)       (0)
  #define __Numba_PyUnicode_GET_LENGTH(u)   PyUnicode_GET_SIZE(u)
  #define __Numba_PyUnicode_READ_CHAR(u, i) ((Py_UCS4)(PyUnicode_AS_UNICODE(u)[i]))
  #define __Numba_PyUnicode_READ(k, d, i)   ((k=k), (Py_UCS4)(((Py_UNICODE*)d)[i]))
#endif

#if PY_MAJOR_VERSION >= 3
  #define PyIntObject                  PyLongObject
  #define PyInt_Type                   PyLong_Type
  #define PyInt_Check(op)              PyLong_Check(op)
  #define PyInt_CheckExact(op)         PyLong_CheckExact(op)
  #define PyInt_FromString             PyLong_FromString
  #define PyInt_FromUnicode            PyLong_FromUnicode
  #define PyInt_FromLong               PyLong_FromLong
  #define PyInt_FromSize_t             PyLong_FromSize_t
  #define PyInt_FromSsize_t            PyLong_FromSsize_t
  #define PyInt_AsLong                 PyLong_AsLong
  #define PyInt_AS_LONG                PyLong_AS_LONG
  #define PyInt_AsSsize_t              PyLong_AsSsize_t
  #define PyInt_AsUnsignedLongMask     PyLong_AsUnsignedLongMask
  #define PyInt_AsUnsignedLongLongMask PyLong_AsUnsignedLongLongMask
#endif

#ifdef __GNUC__
    /* Test for GCC > 2.95 */
    #if __GNUC__ > 2 || (__GNUC__ == 2 && (__GNUC_MINOR__ > 95))
        #define likely(x)   __builtin_expect(!!(x), 1)
        #define unlikely(x) __builtin_expect(!!(x), 0)
    #else /* __GNUC__ > 2 ... */
        #define likely(x)   (x)
        #define unlikely(x) (x)
    #endif /* __GNUC__ > 2 ... */
#else /* __GNUC__ */
    #define likely(x)   (x)
    #define unlikely(x) (x)
#endif /* __GNUC__ */

#define NUMBA_COMPILING_IN_CPYTHON 1

/* End copies from Cython */