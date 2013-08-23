#include <Python.h>
#include <numpy/ndarraytypes.h>
#include <numpy/arrayscalars.h>
#include "generated_conversions.h"
#include "datetime/_datetime.h"
#include "datetime/np_datetime_strings.h"

/* Utilities copied from Cython/Utility/TypeConversion.c */

/////////////// TypeConversions.proto ///////////////

#if PY_MAJOR_VERSION >= 3
  #define PyInt_FromSize_t             PyLong_FromSize_t
  #define PyInt_AsSsize_t              PyLong_AsSsize_t
#endif

/* Type Conversion Predeclarations */

#define __Numba_PyBytes_FromUString(s) PyBytes_FromString((char*)s)
#define __Numba_PyBytes_AsUString(s)   ((unsigned char*) PyBytes_AsString(s))

#define __Numba_Owned_Py_None(b) (Py_INCREF(Py_None), Py_None)
#define __Numba_PyBool_FromLong(b) ((b) ? (Py_INCREF(Py_True), Py_True) : (Py_INCREF(Py_False), Py_False))
static NUMBA_INLINE int __Numba_PyObject_IsTrue(PyObject*);
static NUMBA_INLINE PyObject* __Numba_PyNumber_Int(PyObject* x);

static NUMBA_INLINE Py_ssize_t __Numba_PyIndex_AsSsize_t(PyObject*);
static NUMBA_INLINE PyObject * __Numba_PyInt_FromSize_t(size_t);
static NUMBA_INLINE size_t __Numba_PyInt_AsSize_t(PyObject*);

#if CYTHON_COMPILING_IN_CPYTHON
#define __Numba_PyFloat_AsDouble(x) (PyFloat_CheckExact(x) ? PyFloat_AS_DOUBLE(x) : PyFloat_AsDouble(x))
#else
#define __Numba_PyFloat_AsDouble(x) PyFloat_AsDouble(x)
#endif
#define __Numba_PyFloat_AsFloat(x) ((float) __Numba_PyFloat_AsDouble(x))

/////////////// TypeConversions ///////////////

/* Type Conversion Functions */

/* Note: __Numba_PyObject_IsTrue is written to minimize branching. */
static NUMBA_INLINE int __Numba_PyObject_IsTrue(PyObject* x) {
   int is_true = x == Py_True;
   if (is_true | (x == Py_False) | (x == Py_None)) return is_true;
   else return PyObject_IsTrue(x);
}

static NUMBA_INLINE PyObject* __Numba_PyNumber_Int(PyObject* x) {
  PyNumberMethods *m;
  const char *name = NULL;
  PyObject *res = NULL;
#if PY_VERSION_HEX < 0x03000000
  if (PyInt_Check(x) || PyLong_Check(x))
#else
  if (PyLong_Check(x))
#endif
    return Py_INCREF(x), x;
  m = Py_TYPE(x)->tp_as_number;
#if PY_VERSION_HEX < 0x03000000
  if (m && m->nb_int) {
    name = "int";
    res = PyNumber_Int(x);
  }
  else if (m && m->nb_long) {
    name = "long";
    res = PyNumber_Long(x);
  }
#else
  if (m && m->nb_int) {
    name = "int";
    res = PyNumber_Long(x);
  }
#endif
  if (res) {
#if PY_VERSION_HEX < 0x03000000
    if (!PyInt_Check(res) && !PyLong_Check(res)) {
#else
    if (!PyLong_Check(res)) {
#endif
      PyErr_Format(PyExc_TypeError,
                   "__%s__ returned non-%s (type %.200s)",
                   name, name, Py_TYPE(res)->tp_name);
      Py_DECREF(res);
      return NULL;
    }
  }
  else if (!PyErr_Occurred()) {
    PyErr_SetString(PyExc_TypeError,
                    "an integer is required");
  }
  return res;
}

static NUMBA_INLINE Py_ssize_t __Numba_PyIndex_AsSsize_t(PyObject* b) {
  Py_ssize_t ival;
  PyObject* x = PyNumber_Index(b);
  if (!x) return -1;
  ival = PyInt_AsSsize_t(x);
  Py_DECREF(x);
  return ival;
}

static NUMBA_INLINE PyObject * __Numba_PyInt_FromSize_t(size_t ival) {
#if PY_VERSION_HEX < 0x02050000
   if (ival <= LONG_MAX)
       return PyInt_FromLong((long)ival);
   else {
       unsigned char *bytes = (unsigned char *) &ival;
       int one = 1; int little = (int)*(unsigned char*)&one;
       return _PyLong_FromByteArray(bytes, sizeof(size_t), little, 0);
   }
#else
   return PyInt_FromSize_t(ival);
#endif
}

static NUMBA_INLINE size_t __Numba_PyInt_AsSize_t(PyObject* x) {
   unsigned PY_LONG_LONG val = __Numba_PyInt_AsUnsignedLongLong(x);
   if (unlikely(val == (unsigned PY_LONG_LONG)-1 && PyErr_Occurred())) {
       return (size_t)-1;
   } else if (unlikely(val != (unsigned PY_LONG_LONG)(size_t)val)) {
       PyErr_SetString(PyExc_OverflowError,
                       "value too large to convert to size_t");
       return (size_t)-1;
   }
   return (size_t)val;
}

/////////////// ObjectAsUCS4.proto ///////////////

static NUMBA_INLINE Py_UCS4 __Numba_PyObject_AsPy_UCS4(PyObject*);

/////////////// ObjectAsUCS4 ///////////////

static NUMBA_INLINE Py_UCS4 __Numba_PyObject_AsPy_UCS4(PyObject* x) {
   long ival;
   if (PyUnicode_Check(x)) {
       Py_ssize_t length;
       #if CYTHON_PEP393_ENABLED
       length = PyUnicode_GET_LENGTH(x);
       if (likely(length == 1)) {
           return PyUnicode_READ_CHAR(x, 0);
       }
       #else
       length = PyUnicode_GET_SIZE(x);
       if (likely(length == 1)) {
           return PyUnicode_AS_UNICODE(x)[0];
       }
       #if Py_UNICODE_SIZE == 2
       else if (PyUnicode_GET_SIZE(x) == 2) {
           Py_UCS4 high_val = PyUnicode_AS_UNICODE(x)[0];
           if (high_val >= 0xD800 && high_val <= 0xDBFF) {
               Py_UCS4 low_val = PyUnicode_AS_UNICODE(x)[1];
               if (low_val >= 0xDC00 && low_val <= 0xDFFF) {
                   return 0x10000 + (((high_val & ((1<<10)-1)) << 10) | (low_val & ((1<<10)-1)));
               }
           }
       }
       #endif
       #endif
       PyErr_Format(PyExc_ValueError,
                    "only single character unicode strings can be converted to Py_UCS4, "
                    "got length %" CYTHON_FORMAT_SSIZE_T "d", length);
       return (Py_UCS4)-1;
   }
   ival = __Numba_PyInt_AsSignedLong(x);
   if (unlikely(ival < 0)) {
       if (!PyErr_Occurred())
           PyErr_SetString(PyExc_OverflowError,
                           "cannot convert negative value to Py_UCS4");
       return (Py_UCS4)-1;
   } else if (unlikely(ival > 1114111)) {
       PyErr_SetString(PyExc_OverflowError,
                       "value too large to convert to Py_UCS4");
       return (Py_UCS4)-1;
   }
   return (Py_UCS4)ival;
}

/////////////// ObjectAsPyUnicode.proto ///////////////

static NUMBA_INLINE Py_UNICODE __Numba_PyObject_AsPy_UNICODE(PyObject*);

/////////////// ObjectAsPyUnicode ///////////////

static NUMBA_INLINE Py_UNICODE __Numba_PyObject_AsPy_UNICODE(PyObject* x) {
    long ival;
    #if CYTHON_PEP393_ENABLED
    #if Py_UNICODE_SIZE > 2
    const long maxval = 1114111;
    #else
    const long maxval = 65535;
    #endif
    #else
    static long maxval = 0;
    #endif
    if (PyUnicode_Check(x)) {
        if (unlikely(__Numba_PyUnicode_GET_LENGTH(x) != 1)) {
            PyErr_Format(PyExc_ValueError,
                         "only single character unicode strings can be converted to Py_UNICODE, "
                         "got length %" CYTHON_FORMAT_SSIZE_T "d", __Numba_PyUnicode_GET_LENGTH(x));
            return (Py_UNICODE)-1;
        }
        #if CYTHON_PEP393_ENABLED
        ival = PyUnicode_READ_CHAR(x, 0);
        #else
        return PyUnicode_AS_UNICODE(x)[0];
        #endif
    } else {
        #if !CYTHON_PEP393_ENABLED
        if (unlikely(!maxval))
            maxval = (long)PyUnicode_GetMax();
        #endif
        ival = __Numba_PyInt_AsSignedLong(x);
    }
    if (unlikely(ival < 0)) {
        if (!PyErr_Occurred())
            PyErr_SetString(PyExc_OverflowError,
                            "cannot convert negative value to Py_UNICODE");
        return (Py_UNICODE)-1;
    } else if (unlikely(ival > maxval)) {
        PyErr_SetString(PyExc_OverflowError,
                        "value too large to convert to Py_UNICODE");
        return (Py_UNICODE)-1;
    }
    return (Py_UNICODE)ival;
}

/* End copy from Cython/Utility/TypeConversion.c */
/* --------------------------------------------- */

#define CUTOFF 0x7fffffffL /* 4-byte platform-independent cutoff */

static PyObject *
__Numba_PyInt_FromLongLong(PY_LONG_LONG value)
{
    assert(sizeof(long) >= 4);
    if (value > CUTOFF || value < -CUTOFF) {
        return PyLong_FromLongLong(value);
    }
    return PyInt_FromLong(value);
}

static PyObject *
__Numba_PyInt_FromUnsignedLongLong(PY_LONG_LONG value)
{
    assert(sizeof(long) >= 4);
    if (value > CUTOFF) {
        return PyLong_FromUnsignedLongLong(value);
    }
    return PyInt_FromLong((long) value);
}

#include "generated_conversions.c"

/* Export all utilities */
static NUMBA_INLINE int __Numba_PyObject_IsTrue(PyObject*);
static NUMBA_INLINE PyObject* __Numba_PyNumber_Int(PyObject* x);

static NUMBA_INLINE Py_ssize_t __Numba_PyIndex_AsSsize_t(PyObject*);
static NUMBA_INLINE PyObject * __Numba_PyInt_FromSize_t(size_t);
static NUMBA_INLINE size_t __Numba_PyInt_AsSize_t(PyObject*);

static npy_datetimestruct iso_datetime2npydatetime(char *datetime_string)
{
    npy_datetimestruct out;
    npy_bool out_local;
    NPY_DATETIMEUNIT out_bestunit;
    npy_bool out_special;

    parse_iso_8601_datetime(datetime_string, strlen(datetime_string), -1, NPY_SAME_KIND_CASTING,
        &out, &out_local, &out_bestunit, &out_special);

    return out;
}

static npy_int64 iso_datetime2year(char *datetime_string)
{
    npy_datetimestruct out;
    out = iso_datetime2npydatetime(datetime_string);
    return out.year;
}

static npy_int64 iso_datetime2month(char *datetime_string)
{
    npy_datetimestruct out;
    out = iso_datetime2npydatetime(datetime_string);
    return out.month;
}

static npy_int64 iso_datetime2day(char *datetime_string)
{
    npy_datetimestruct out;
    out = iso_datetime2npydatetime(datetime_string);
    return out.day;
}

static npy_int64 iso_datetime2hour(char *datetime_string)
{
    npy_datetimestruct out;
    out = iso_datetime2npydatetime(datetime_string);
    return out.hour;
}

static npy_int64 iso_datetime2min(char *datetime_string)
{
    npy_datetimestruct out;
    out = iso_datetime2npydatetime(datetime_string);
    return out.min;
}

static npy_int64 iso_datetime2sec(char *datetime_string)
{
    npy_datetimestruct out;
    out = iso_datetime2npydatetime(datetime_string);
    return out.sec;
}


static npy_datetimestruct pydatetime2npydatetime_struct(PyObject *object)
{
    PyArray_DatetimeMetaData meta;
    npy_datetime out;
    npy_datetimestruct out2;
    int result;

    meta.base = -1;
    convert_pyobject_to_datetime(&meta, object, NPY_SAME_KIND_CASTING, &out);
    convert_datetime_to_datetimestruct(&meta, out, &out2);
    return out2;
}

static npy_int64 pydatetime2year(PyObject *object)
{
    npy_datetimestruct out = pydatetime2npydatetime_struct(object);
    return out.year;
}

static npy_int32 pydatetime2month(PyObject *object)
{
    npy_datetimestruct out = pydatetime2npydatetime_struct(object);
    return out.month;
}

static npy_int32 pydatetime2day(PyObject *object)
{
    npy_datetimestruct out = pydatetime2npydatetime_struct(object);
    return out.day;
}

static npy_int32 pydatetime2hour(PyObject *object)
{
    npy_datetimestruct out = pydatetime2npydatetime_struct(object);
    return out.hour;
}

static npy_int32 pydatetime2min(PyObject *object)
{
    npy_datetimestruct out = pydatetime2npydatetime_struct(object);
    return out.min;
}

static npy_int32 pydatetime2sec(PyObject *object)
{
    npy_datetimestruct out = pydatetime2npydatetime_struct(object);
    return out.sec;
}

static npy_int32 pydatetime2usec(PyObject *object)
{
    npy_datetimestruct out = pydatetime2npydatetime_struct(object);
    return out.us;
}

static npy_int32 pydatetime2psec(PyObject *object)
{
    npy_datetimestruct out = pydatetime2npydatetime_struct(object);
    return out.ps;
}

static npy_int32 pydatetime2asec(PyObject *object)
{
    npy_datetimestruct out = pydatetime2npydatetime_struct(object);
    return out.as;
}

static PyObject* primitive2pydatetime(
    npy_int64 year,
    npy_int32 month,
    npy_int32 day,
    npy_int32 hour,
    npy_int32 min,
    npy_int32 sec)
{
    PyObject *result = PyDateTime_FromDateAndTime(year, month, day,
        hour, min, sec, 0);
    Py_INCREF(result);
    return result;
}

static PyObject* primitive2numpydatetime(
    npy_int64 year,
    npy_int32 month,
    npy_int32 day,
    npy_int32 hour,
    npy_int32 min,
    npy_int32 sec,
    PyDatetimeScalarObject *scalar)
{
    npy_datetimestruct input;
    npy_datetime output = 0;
    PyArray_DatetimeMetaData new_meta;

    memset(&input, 0, sizeof(input));
    input.year = year;
    input.month = month;
    input.day = day;
    input.hour = hour;
    input.min = min;
    input.sec = sec;

    new_meta.base = lossless_unit_from_datetimestruct(&input);
    new_meta.num = 1;

    if (convert_datetimestruct_to_datetime(&new_meta, &input, &output) < 0) {
        return NULL;
    }
    
    scalar->obval = output;
    scalar->obmeta.base = new_meta.base;
    scalar->obmeta.num = new_meta.num;
 
    return (PyObject*)scalar;
}


static npy_datetimestruct numpydatetime2npydatetime_struct(PyObject *numpy_datetime)
{
    PyArray_DatetimeMetaData meta;
    npy_datetime value;
    npy_datetimestruct out;

    meta.base = ((PyDatetimeScalarObject*)numpy_datetime)->obmeta.base;
    meta.num = ((PyDatetimeScalarObject*)numpy_datetime)->obmeta.num;
    value = ((PyDatetimeScalarObject*)numpy_datetime)->obval;
    memset(&out, 0, sizeof(npy_datetimestruct));

    convert_datetime_to_datetimestruct(&meta, value, &out);

    return out;
}

static npy_int64 numpydatetime2year(PyObject *numpy_datetime)
{
    npy_datetimestruct out = numpydatetime2npydatetime_struct(numpy_datetime);
    return out.year;
}

static npy_int64 numpydatetime2month(PyObject *numpy_datetime)
{
    npy_datetimestruct out = numpydatetime2npydatetime_struct(numpy_datetime);
    return out.month;
}

static npy_int64 numpydatetime2day(PyObject *numpy_datetime)
{
    npy_datetimestruct out = numpydatetime2npydatetime_struct(numpy_datetime);
    return out.day;
}

static npy_int64 numpydatetime2hour(PyObject *numpy_datetime)
{
    npy_datetimestruct out = numpydatetime2npydatetime_struct(numpy_datetime);
    return out.hour;
}

static npy_int64 numpydatetime2min(PyObject *numpy_datetime)
{
    npy_datetimestruct out = numpydatetime2npydatetime_struct(numpy_datetime);
    return out.min;
}

static npy_int64 numpydatetime2sec(PyObject *numpy_datetime)
{
    npy_datetimestruct out = numpydatetime2npydatetime_struct(numpy_datetime);
    return out.sec;
}


static int
export_type_conversion(PyObject *module)
{
    EXPORT_FUNCTION(__Numba_PyInt_AsSignedChar, module, error)
    EXPORT_FUNCTION(__Numba_PyInt_AsUnsignedChar, module, error)
    EXPORT_FUNCTION(__Numba_PyInt_AsSignedShort, module, error)
    EXPORT_FUNCTION(__Numba_PyInt_AsUnsignedShort, module, error)
    EXPORT_FUNCTION(__Numba_PyInt_AsSignedInt, module, error)
    EXPORT_FUNCTION(__Numba_PyInt_AsUnsignedInt, module, error)
    EXPORT_FUNCTION(__Numba_PyInt_AsSignedLong, module, error)
    EXPORT_FUNCTION(__Numba_PyInt_AsUnsignedLong, module, error)
    EXPORT_FUNCTION(__Numba_PyInt_AsSignedLongLong, module, error)
    EXPORT_FUNCTION(__Numba_PyInt_AsUnsignedLongLong, module, error)

    EXPORT_FUNCTION(__Numba_PyIndex_AsSsize_t, module, error);
    EXPORT_FUNCTION(__Numba_PyInt_FromSize_t, module, error);
    
    EXPORT_FUNCTION(pydatetime2year, module, error);
    EXPORT_FUNCTION(pydatetime2month, module, error);
    EXPORT_FUNCTION(pydatetime2day, module, error);
    EXPORT_FUNCTION(pydatetime2hour, module, error);
    EXPORT_FUNCTION(pydatetime2min, module, error);
    EXPORT_FUNCTION(pydatetime2sec, module, error);
    EXPORT_FUNCTION(pydatetime2usec, module, error);
    EXPORT_FUNCTION(pydatetime2psec, module, error);
    EXPORT_FUNCTION(pydatetime2asec, module, error);
    EXPORT_FUNCTION(primitive2pydatetime, module, error);
    EXPORT_FUNCTION(primitive2numpydatetime, module, error);
    EXPORT_FUNCTION(iso_datetime2year, module, error);
    EXPORT_FUNCTION(iso_datetime2month, module, error);
    EXPORT_FUNCTION(iso_datetime2day, module, error);
    EXPORT_FUNCTION(iso_datetime2hour, module, error);
    EXPORT_FUNCTION(iso_datetime2min, module, error);
    EXPORT_FUNCTION(iso_datetime2sec, module, error);
    EXPORT_FUNCTION(numpydatetime2year, module, error);
    EXPORT_FUNCTION(numpydatetime2month, module, error);
    EXPORT_FUNCTION(numpydatetime2day, module, error);
    EXPORT_FUNCTION(numpydatetime2hour, module, error);
    EXPORT_FUNCTION(numpydatetime2min, module, error);
    EXPORT_FUNCTION(numpydatetime2sec, module, error);

    EXPORT_FUNCTION(__Numba_PyInt_FromLongLong, module, error);
    EXPORT_FUNCTION(__Numba_PyInt_FromUnsignedLongLong, module, error);

    return 0;
error:
    return -1;
}

