/*
 * Helper functions used by Numba at runtime.
 * This C file is meant to be included after defining the
 * NUMBA_EXPORT_FUNC() and NUMBA_EXPORT_DATA() macros.
 */

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
#include <numpy/npy_math.h>

#include "_arraystruct.h"

/*
 * Other helpers.
 */


/* Fix fmod() and fmodf() for windows x64 VC 9.0 (VS 2008)
https://support.microsoft.com/en-us/kb/982107
*/
static  void (*fnclex)(void) = NULL;

NUMBA_EXPORT_FUNC(double)
numba_fixed_fmod(double x, double y){
    fnclex();  /* no inline asm in x64 =( */
    return fmod(x, y);
}

NUMBA_EXPORT_FUNC(float)
numba_fixed_fmodf(float x, float y) {
    fnclex();  /* no inline asm in x64 =( */
    return fmodf(x, y);
}

NUMBA_EXPORT_FUNC(void)
numba_set_fnclex(void *fn){
    fnclex = fn;
}

/* provide 64-bit division function to 32-bit platforms */
NUMBA_EXPORT_FUNC(int64_t)
numba_sdiv(int64_t a, int64_t b) {
    return a / b;
}

NUMBA_EXPORT_FUNC(uint64_t)
numba_udiv(uint64_t a, uint64_t b) {
    return a / b;
}

/* provide 64-bit remainder function to 32-bit platforms */
NUMBA_EXPORT_FUNC(int64_t)
numba_srem(int64_t a, int64_t b) {
    return a % b;
}

NUMBA_EXPORT_FUNC(uint64_t)
numba_urem(uint64_t a, uint64_t b) {
    return a % b;
}

/* provide frexp and ldexp; these wrappers deal with special cases
 * (zero, nan, infinity) directly, to sidestep platform differences.
 */
NUMBA_EXPORT_FUNC(double)
numba_frexp(double x, int *exp)
{
    if (!Py_IS_FINITE(x) || !x)
        *exp = 0;
    else
        x = frexp(x, exp);
    return x;
}

NUMBA_EXPORT_FUNC(float)
numba_frexpf(float x, int *exp)
{
    if (Py_IS_NAN(x) || Py_IS_INFINITY(x) || !x)
        *exp = 0;
    else
        x = frexpf(x, exp);
    return x;
}

NUMBA_EXPORT_FUNC(double)
numba_ldexp(double x, int exp)
{
    if (Py_IS_FINITE(x) && x && exp)
        x = ldexp(x, exp);
    return x;
}

NUMBA_EXPORT_FUNC(float)
numba_ldexpf(float x, int exp)
{
    if (Py_IS_FINITE(x) && x && exp)
        x = ldexpf(x, exp);
    return x;
}

/* provide complex power */
NUMBA_EXPORT_FUNC(void)
numba_cpow(Py_complex *a, Py_complex *b, Py_complex *out) {
    errno = 0;
    *out = _Py_c_pow(*a, *b);
    if (errno == EDOM) {
        /* _Py_c_pow() doesn't bother returning the right value
           in this case, as Python raises ZeroDivisionError */
        out->real = out->imag = Py_NAN;
    }
}

NUMBA_EXPORT_FUNC(void)
numba_cpowf(npy_cfloat *a, npy_cfloat *b, npy_cfloat *out) {
    Py_complex _a, _b, _out;
    _a.real = npy_crealf(*a);
    _a.imag = npy_cimagf(*a);
    _b.real = npy_crealf(*b);
    _b.imag = npy_cimagf(*b);
    numba_cpow(&_a, &_b, &_out);
    *out = npy_cpackf((float) _out.real, (float) _out.imag);
}

/* C99 math functions: redirect to system implementations
   (but see _math_c99.h for Windows) */

NUMBA_EXPORT_FUNC(double)
numba_gamma(double x)
{
    return tgamma(x);
}

NUMBA_EXPORT_FUNC(float)
numba_gammaf(float x)
{
    return tgammaf(x);
}

NUMBA_EXPORT_FUNC(double)
numba_lgamma(double x)
{
    return lgamma(x);
}

NUMBA_EXPORT_FUNC(float)
numba_lgammaf(float x)
{
    return lgammaf(x);
}

NUMBA_EXPORT_FUNC(double)
numba_erf(double x)
{
    return erf(x);
}

NUMBA_EXPORT_FUNC(float)
numba_erff(float x)
{
    return erff(x);
}

NUMBA_EXPORT_FUNC(double)
numba_erfc(double x)
{
    return erfc(x);
}

NUMBA_EXPORT_FUNC(float)
numba_erfcf(float x)
{
    return erfcf(x);
}

/* Note npy_signbit() is actually a polymorphic macro */
NUMBA_EXPORT_FUNC(int)
numba_signbitf(float a)
{
    return npy_signbit(a);
}

NUMBA_EXPORT_FUNC(int)
numba_signbit(npy_double a)
{
    return npy_signbit(a);
}


/* Unpack any Python complex-like object into a Py_complex structure */
NUMBA_EXPORT_FUNC(int)
numba_complex_adaptor(PyObject* obj, Py_complex *out) {
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
NUMBA_EXPORT_FUNC(void *)
numba_extract_record_data(PyObject *recordobj, Py_buffer *pbuf) {
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
NUMBA_EXPORT_FUNC(PyObject *)
numba_recreate_record(void *pdata, int size, PyObject *dtype) {
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

NUMBA_EXPORT_FUNC(int)
numba_adapt_ndarray(PyObject *obj, arystruct_t* arystruct) {
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

NUMBA_EXPORT_FUNC(int)
numba_get_buffer(PyObject *obj, Py_buffer *buf)
{
    /* Ask for shape and strides, but no suboffsets */
    return PyObject_GetBuffer(obj, buf, PyBUF_RECORDS_RO);
}

NUMBA_EXPORT_FUNC(void)
numba_adapt_buffer(Py_buffer *buf, arystruct_t *arystruct)
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

NUMBA_EXPORT_FUNC(void)
numba_release_buffer(Py_buffer *buf)
{
    PyBuffer_Release(buf);
}

NUMBA_EXPORT_FUNC(PyObject *)
numba_ndarray_new(int nd,
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


/*
 * Handle reshaping of zero-sized array.
 * See numba_attempt_nocopy_reshape() below.
 */
static int
nocopy_empty_reshape(npy_intp nd, const npy_intp *dims, const npy_intp *strides,
                     npy_intp newnd, const npy_intp *newdims,
                     npy_intp *newstrides, npy_intp itemsize,
                     int is_f_order)
{
    int i;
    /* Just make the strides vaguely reasonable
     * (they can have any value in theory).
     */
    for (i = 0; i < newnd; i++)
        newstrides[i] = itemsize;
    return 1;  /* reshape successful */
}

/*
 * Straight from Numpy's _attempt_nocopy_reshape()
 * (np/core/src/multiarray/shape.c).
 * Attempt to reshape an array without copying data
 *
 * This function should correctly handle all reshapes, including
 * axes of length 1. Zero strides should work but are untested.
 *
 * If a copy is needed, returns 0
 * If no copy is needed, returns 1 and fills `npy_intp *newstrides`
 *     with appropriate strides
 */

NUMBA_EXPORT_FUNC(int)
numba_attempt_nocopy_reshape(npy_intp nd, const npy_intp *dims, const npy_intp *strides,
                             npy_intp newnd, const npy_intp *newdims,
                             npy_intp *newstrides, npy_intp itemsize,
                             int is_f_order)
{
    int oldnd;
    npy_intp olddims[NPY_MAXDIMS];
    npy_intp oldstrides[NPY_MAXDIMS];
    npy_intp np, op, last_stride;
    int oi, oj, ok, ni, nj, nk;

    oldnd = 0;
    /*
     * Remove axes with dimension 1 from the old array. They have no effect
     * but would need special cases since their strides do not matter.
     */
    for (oi = 0; oi < nd; oi++) {
        if (dims[oi]!= 1) {
            olddims[oldnd] = dims[oi];
            oldstrides[oldnd] = strides[oi];
            oldnd++;
        }
    }

    np = 1;
    for (ni = 0; ni < newnd; ni++) {
        np *= newdims[ni];
    }
    op = 1;
    for (oi = 0; oi < oldnd; oi++) {
        op *= olddims[oi];
    }
    if (np != op) {
        /* different total sizes; no hope */
        return 0;
    }

    if (np == 0) {
        /* the Numpy code does not handle 0-sized arrays */
        return nocopy_empty_reshape(nd, dims, strides,
                                    newnd, newdims, newstrides,
                                    itemsize, is_f_order);
    }

    /* oi to oj and ni to nj give the axis ranges currently worked with */
    oi = 0;
    oj = 1;
    ni = 0;
    nj = 1;
    while (ni < newnd && oi < oldnd) {
        np = newdims[ni];
        op = olddims[oi];

        while (np != op) {
            if (np < op) {
                /* Misses trailing 1s, these are handled later */
                np *= newdims[nj++];
            } else {
                op *= olddims[oj++];
            }
        }

        /* Check whether the original axes can be combined */
        for (ok = oi; ok < oj - 1; ok++) {
            if (is_f_order) {
                if (oldstrides[ok+1] != olddims[ok]*oldstrides[ok]) {
                     /* not contiguous enough */
                    return 0;
                }
            }
            else {
                /* C order */
                if (oldstrides[ok] != olddims[ok+1]*oldstrides[ok+1]) {
                    /* not contiguous enough */
                    return 0;
                }
            }
        }

        /* Calculate new strides for all axes currently worked with */
        if (is_f_order) {
            newstrides[ni] = oldstrides[oi];
            for (nk = ni + 1; nk < nj; nk++) {
                newstrides[nk] = newstrides[nk - 1]*newdims[nk - 1];
            }
        }
        else {
            /* C order */
            newstrides[nj - 1] = oldstrides[oj - 1];
            for (nk = nj - 1; nk > ni; nk--) {
                newstrides[nk - 1] = newstrides[nk]*newdims[nk];
            }
        }
        ni = nj++;
        oi = oj++;
    }

    /*
     * Set strides corresponding to trailing 1s of the new shape.
     */
    if (ni >= 1) {
        last_stride = newstrides[ni - 1];
    }
    else {
        last_stride = itemsize;
    }
    if (is_f_order) {
        last_stride *= newdims[ni - 1];
    }
    for (nk = ni; nk < newnd; nk++) {
        newstrides[nk] = last_stride;
    }

    return 1;
}

/*
 * Cython utilities.
 */

/* Fetch the address of the given function, as exposed by
   a cython module */
static void *
import_cython_function(const char *module_name, const char *function_name)
{
    PyObject *module, *capi, *cobj;
    void *res = NULL;
    const char *capsule_name;

    module = PyImport_ImportModule(module_name);
    if (module == NULL)
        return NULL;
    capi = PyObject_GetAttrString(module, "__pyx_capi__");
    Py_DECREF(module);
    if (capi == NULL)
        return NULL;
    cobj = PyMapping_GetItemString(capi, function_name);
    Py_DECREF(capi);
    if (cobj == NULL) {
	PyErr_Clear();
	PyErr_Format(PyExc_ValueError,
		     "No function '%s' found in __pyx_capi__ of '%s'",
		     function_name, module_name);
        return NULL;
    }
    /* 2.7+ => Cython exports a PyCapsule */
    capsule_name = PyCapsule_GetName(cobj);
    if (capsule_name != NULL) {
        res = PyCapsule_GetPointer(cobj, capsule_name);
    }
    Py_DECREF(cobj);
    return res;
}

NUMBA_EXPORT_FUNC(PyObject *)
_numba_import_cython_function(PyObject *self, PyObject *args)
{
    const char *module_name;
    const char *function_name;
    void *p = NULL;
    PyObject *res;

    if (!PyArg_ParseTuple(args, "ss", &module_name, &function_name)) {
	return NULL;
    }
    p = import_cython_function(module_name, function_name);
    if (p == NULL) {
	return NULL;
    }
    res = PyLong_FromVoidPtr(p);
    if (res == NULL) {
      PyErr_SetString(PyExc_RuntimeError,
		      "Could not convert function address to int");
      return NULL;
    }
    return res;
}

/* We use separate functions for datetime64 and timedelta64, to ensure
 * proper type checking.
 */
NUMBA_EXPORT_FUNC(npy_int64)
numba_extract_np_datetime(PyObject *td)
{
    if (!PyArray_IsScalar(td, Datetime)) {
        PyErr_SetString(PyExc_TypeError,
                        "expected a numpy.datetime64 object");
        return -1;
    }
    return PyArrayScalar_VAL(td, Timedelta);
}

NUMBA_EXPORT_FUNC(npy_int64)
numba_extract_np_timedelta(PyObject *td)
{
    if (!PyArray_IsScalar(td, Timedelta)) {
        PyErr_SetString(PyExc_TypeError,
                        "expected a numpy.timedelta64 object");
        return -1;
    }
    return PyArrayScalar_VAL(td, Timedelta);
}

NUMBA_EXPORT_FUNC(PyObject *)
numba_create_np_datetime(npy_int64 value, int unit_code)
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

NUMBA_EXPORT_FUNC(PyObject *)
numba_create_np_timedelta(npy_int64 value, int unit_code)
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

NUMBA_EXPORT_FUNC(uint64_t)
numba_fptoui(double x) {
    /* First cast to signed int of the full width to make sure sign extension
       happens (this can make a difference on some platforms...). */
    return (uint64_t) (int64_t) x;
}

NUMBA_EXPORT_FUNC(uint64_t)
numba_fptouif(float x) {
    return (uint64_t) (int64_t) x;
}

NUMBA_EXPORT_FUNC(void)
numba_gil_ensure(PyGILState_STATE *state) {
    *state = PyGILState_Ensure();
}

NUMBA_EXPORT_FUNC(void)
numba_gil_release(PyGILState_STATE *state) {
    PyGILState_Release(*state);
}

NUMBA_EXPORT_FUNC(PyObject *)
numba_py_type(PyObject *obj) {
    return (PyObject *) Py_TYPE(obj);
}


/*
 * Functions for tagging an arbitrary Python object with an arbitrary pointer.
 * These functions make strong lifetime assumptions, see below.
 */

static PyObject *private_data_dict = NULL;

static PyObject *
_get_private_data_dict(void)
{
    if (private_data_dict == NULL)
        private_data_dict = PyDict_New();
    return private_data_dict;
}

NUMBA_EXPORT_FUNC(void)
numba_set_pyobject_private_data(PyObject *obj, void *ptr)
{
    PyObject *dct = _get_private_data_dict();
    /* This assumes the reference to setobj is kept alive until the
       call to numba_reset_set_private_data()! */
    PyObject *key = PyLong_FromVoidPtr((void *) obj);
    PyObject *value = PyLong_FromVoidPtr(ptr);

    if (!dct || !value || !key)
        goto error;
    if (PyDict_SetItem(dct, key, value))
        goto error;
    Py_DECREF(key);
    Py_DECREF(value);
    return;

error:
    Py_FatalError("unable to set private data");
}

NUMBA_EXPORT_FUNC(void *)
numba_get_pyobject_private_data(PyObject *obj)
{
    PyObject *dct = _get_private_data_dict();
    PyObject *value, *key = PyLong_FromVoidPtr((void *) obj);
    void *ptr;
    if (!dct || !key)
        goto error;

    value = PyDict_GetItem(dct, key);
    Py_DECREF(key);
    if (!value)
        return NULL;
    else {
        ptr = PyLong_AsVoidPtr(value);
        if (ptr == NULL && PyErr_Occurred())
            goto error;
        return ptr;
    }

error:
    Py_FatalError("unable to get private data");
    return NULL;
}

NUMBA_EXPORT_FUNC(void)
numba_reset_pyobject_private_data(PyObject *obj)
{
    PyObject *dct = _get_private_data_dict();
    PyObject *key = PyLong_FromVoidPtr((void *) obj);

    if (!key)
        goto error;
    if (PyDict_DelItem(dct, key))
        PyErr_Clear();
    Py_DECREF(key);
    return;

error:
    Py_FatalError("unable to reset private data");
}

NUMBA_EXPORT_FUNC(int)
numba_unpack_slice(PyObject *obj,
                   Py_ssize_t *start, Py_ssize_t *stop, Py_ssize_t *step)
{
    PySliceObject *slice = (PySliceObject *) obj;
    if (!PySlice_Check(obj)) {
        PyErr_Format(PyExc_TypeError,
                     "Expected a slice object, got '%s'",
                     Py_TYPE(slice)->tp_name);
        return -1;
    }
#define FETCH_MEMBER(NAME, DEFAULT)                             \
    if (slice->NAME != Py_None) {                               \
        Py_ssize_t v = PyNumber_AsSsize_t(slice->NAME,          \
                                          PyExc_OverflowError); \
        if (v == -1 && PyErr_Occurred())                        \
            return -1;                                          \
        *NAME = v;                                              \
    }                                                           \
    else {                                                      \
        *NAME = DEFAULT;                                        \
    }
    FETCH_MEMBER(step, 1)
    FETCH_MEMBER(stop, (*step > 0) ? PY_SSIZE_T_MAX : PY_SSIZE_T_MIN)
    FETCH_MEMBER(start, (*step > 0) ? 0 : PY_SSIZE_T_MAX)
    return 0;

#undef FETCH_MEMBER
}

NUMBA_EXPORT_FUNC(int)
numba_fatal_error(void)
{
    PyGILState_Ensure();
#if PY_MAJOR_VERSION < 3
    /* Py_FatalError doesn't print the current error on Python 2, do it
       ourselves. */
    if (PyErr_Occurred())
        PyErr_Print();
#endif
    Py_FatalError("in Numba-compiled function");
    return 0; /* unreachable */
}

/* Insert a frame into the traceback for (funcname, filename, lineno). */
/* This function is CPython's _PyTraceback_Add, renamed, see:
 * https://github.com/python/cpython/blob/d545869d084e70d4838310e79b52a25a72a1ca56/Python/traceback.c#L246
 * and modified for Python 2.x based on
 * https://github.com/python/cpython/blob/2e1a34025cde19bddf12a2eac8fedb6afcca8339/Modules/_ctypes/callbacks.c#L151-L174
 */
static void traceback_add(const char *funcname, const char *filename, int lineno)
{
    PyObject *globals = NULL;
    PyCodeObject *code = NULL;
    PyFrameObject *frame = NULL;
    PyObject *exc, *val, *tb;

    /* Save and clear the current exception. Python functions must not be
       called with an exception set. Calling Python functions happens when
       the codec of the filesystem encoding is implemented in pure Python. */
    PyErr_Fetch(&exc, &val, &tb);

    globals = PyDict_New();
    if (!globals)
        goto error;
    code = PyCode_NewEmpty(filename, funcname, lineno);
    if (!code) {
        goto error;
    }
    frame = PyFrame_New(PyThreadState_Get(), code, globals, NULL);
    Py_DECREF(globals);
    Py_DECREF(code);
    if (!frame)
        goto error;
    frame->f_lineno = lineno;

    PyErr_Restore(exc, val, tb);
    PyTraceBack_Here(frame);
    Py_DECREF(frame);
    return;

error:
#if PY_MAJOR_VERSION >= 3
    _PyErr_ChainExceptions(exc, val, tb);
#else
    Py_XDECREF(globals);
    Py_XDECREF(code);
    Py_XDECREF(frame);
#endif
}

/* Logic for raising an arbitrary object.  Adapted from CPython's ceval.c.
   This *consumes* a reference count to its argument. */
NUMBA_EXPORT_FUNC(int)
numba_do_raise(PyObject *exc_packed)
{
    PyObject *exc = NULL, *type = NULL, *value = NULL, *loc = NULL;
    const char *function_name_str = NULL, *filename_str = NULL;
    PyObject *function_name = NULL, *filename = NULL, *lineno = NULL;
    Py_ssize_t pos;

    /* We support the following forms of raise:
       raise
       raise <instance>
       raise <type> */

    /* could be a tuple from npm (some exc like thing, args, location) */
    if (PyTuple_CheckExact(exc_packed)) {
        /* Unpack a (class/inst/tuple, arguments, location) tuple. */
        if (!PyArg_ParseTuple(exc_packed, "OOO", &exc, &value, &loc)) {
            Py_DECREF(exc_packed);
            goto raise_error;
        }

        if (exc == Py_None) {
            /* Reraise */
            PyThreadState *tstate = PyThreadState_GET();
            PyObject *tb;
#if (PY_MAJOR_VERSION >= 3) && (PY_MINOR_VERSION >= 7)
            _PyErr_StackItem *tstate_exc = tstate->exc_info;
#else
            PyThreadState *tstate_exc = tstate;
#endif
            Py_DECREF(exc_packed);
            type = tstate_exc->exc_type;
            value = tstate_exc->exc_value;
            tb = tstate_exc->exc_traceback;
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

        /* the unpacked exc should be a class, value and loc are set from above
         */
        Py_XINCREF(value);
        Py_XINCREF(loc);
        if (PyExceptionClass_Check(exc)) {
            /* It is a class, type used here just as a tmp var */
            type = PyObject_CallObject(exc, NULL);
            if (type == NULL)
                goto raise_error;
            if (!PyExceptionInstance_Check(type)) {
                PyErr_SetString(PyExc_TypeError,
                                "exceptions must derive from BaseException");
                goto raise_error;
            }
            /* all ok, set type to the exc */
            Py_DECREF(type);
            type = exc;
        } else {
            /* this should be unreachable as typing should catch it */
            /* Not something you can raise.  You get an exception
            anyway, just not what you specified :-) */
            Py_DECREF(exc_packed);
            PyErr_SetString(PyExc_TypeError,
                            "exceptions must derive from BaseException");
            goto raise_error;
        }

        /* for injecting a frame into the traceback with location info in it */
        if (loc != Py_None && PyTuple_Check(loc)) {
            pos = 0;
            function_name = PyTuple_GET_ITEM(loc, pos);
            function_name_str = PyString_AsString(function_name);
            pos = 1;
            filename = PyTuple_GET_ITEM(loc, pos);
            filename_str = PyString_AsString(filename);
            pos = 2;
            lineno = PyTuple_GET_ITEM(loc, pos);
        }
        /* as this branch is exited:
         * - type should be an exception class
         * - value should be the args for the exception class instantiation
         * - loc should be the location information (or None)
         */
    } else {  /* could be a reraise or an exception from objmode */
        exc = exc_packed;
        if (exc == Py_None) {
            /* Reraise */
            PyThreadState *tstate = PyThreadState_GET();
            PyObject *tb;
#if (PY_MAJOR_VERSION >= 3) && (PY_MINOR_VERSION >= 7)
            _PyErr_StackItem *tstate_exc = tstate->exc_info;
#else
            PyThreadState *tstate_exc = tstate;
#endif
            Py_DECREF(exc);
            type = tstate_exc->exc_type;
            value = tstate_exc->exc_value;
            tb = tstate_exc->exc_traceback;
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

        /* exc should be an exception class or an instance of an exception */
        if (PyExceptionClass_Check(exc)) {
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
            Py_DECREF(exc); // exc points to exc_packed
            PyErr_SetString(PyExc_TypeError,
                            "exceptions must derive from BaseException");
            goto raise_error;
        }
    }

    PyErr_SetObject(type, value);

    /* instance is instantiated, if loc is present add a frame for it */
    if(loc && loc != Py_None)
    {
        traceback_add(function_name_str, filename_str, \
                      (int)PyLong_AsLong(lineno));
    }

    /* PyErr_SetObject incref's its arguments */
    Py_XDECREF(value);
    Py_XDECREF(type);
    return 0;

raise_error:
    Py_XDECREF(value);
    Py_XDECREF(type);
    return 0;
}

NUMBA_EXPORT_FUNC(PyObject *)
numba_unpickle(const char *data, int n)
{
    PyObject *buf, *obj;
    static PyObject *loads;

    /* Caching the pickle.loads function shaves a couple µs here. */
    if (loads == NULL) {
        PyObject *picklemod;
#if PY_MAJOR_VERSION >= 3
        picklemod = PyImport_ImportModule("pickle");
#else
        picklemod = PyImport_ImportModule("cPickle");
#endif
        if (picklemod == NULL)
            return NULL;
        loads = PyObject_GetAttrString(picklemod, "loads");
        Py_DECREF(picklemod);
        if (loads == NULL)
            return NULL;
    }

    buf = PyBytes_FromStringAndSize(data, n);
    if (buf == NULL)
        return NULL;
    obj = PyObject_CallFunctionObjArgs(loads, buf, NULL);
    Py_DECREF(buf);
    return obj;
}

/*
 * Unicode helpers
 */

NUMBA_EXPORT_FUNC(void *)
numba_extract_unicode(PyObject *obj, Py_ssize_t *length, int *kind) {
#if (PY_MAJOR_VERSION >= 3) && (PY_MINOR_VERSION >= 3)
    if (!PyUnicode_READY(obj)) {
        *length = PyUnicode_GET_LENGTH(obj);
        *kind = PyUnicode_KIND(obj);
        return PyUnicode_DATA(obj);
    } else {
        return NULL;
    }
#else
    /* this function only works in Python 3 */
    return NULL;
#endif
}

/*
 * Define bridge for all math functions
 */

#define MATH_UNARY(F, R, A) \
    NUMBA_EXPORT_FUNC(R) numba_##F(A a) { return F(a); }
#define MATH_BINARY(F, R, A, B) \
    NUMBA_EXPORT_FUNC(R) numba_##F(A a, B b) { return F(a, b); }

#include "mathnames.h"

#undef MATH_UNARY
#undef MATH_BINARY

/*
 * BLAS and LAPACK wrappers
 */

#include "_lapack.c"

/*
 * PRNG support
 */

#include "_random.c"
