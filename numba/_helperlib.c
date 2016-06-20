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

#include "_arraystruct.h"

/*
 * PRNG support.
 */

/* Magic Mersenne Twister constants */
#define MT_N 624
#define MT_M 397
#define MT_MATRIX_A 0x9908b0dfU
#define MT_UPPER_MASK 0x80000000U
#define MT_LOWER_MASK 0x7fffffffU

/* unsigned int is sufficient on modern machines as we only need 32 bits */
typedef struct {
    int index;
    unsigned int mt[MT_N];
    int has_gauss;
    double gauss;
} rnd_state_t;

NUMBA_EXPORT_DATA(rnd_state_t) numba_py_random_state;
NUMBA_EXPORT_DATA(rnd_state_t) numba_np_random_state;

/* Some code portions below from CPython's _randommodule.c, some others
   from Numpy's and Jean-Sebastien Roy's randomkit.c. */

NUMBA_EXPORT_FUNC(void)
numba_rnd_shuffle(rnd_state_t *state)
{
    int i;
    unsigned int y;

    for (i = 0; i < MT_N - MT_M; i++) {
        y = (state->mt[i] & MT_UPPER_MASK) | (state->mt[i+1] & MT_LOWER_MASK);
        state->mt[i] = state->mt[i+MT_M] ^ (y >> 1) ^
                       (-(int) (y & 1) & MT_MATRIX_A);
    }
    for (; i < MT_N - 1; i++) {
        y = (state->mt[i] & MT_UPPER_MASK) | (state->mt[i+1] & MT_LOWER_MASK);
        state->mt[i] = state->mt[i+(MT_M-MT_N)] ^ (y >> 1) ^
                       (-(int) (y & 1) & MT_MATRIX_A);
    }
    y = (state->mt[MT_N - 1] & MT_UPPER_MASK) | (state->mt[0] & MT_LOWER_MASK);
    state->mt[MT_N - 1] = state->mt[MT_M - 1] ^ (y >> 1) ^
                          (-(int) (y & 1) & MT_MATRIX_A);
}

/* Initialize mt[] with an integer seed */
NUMBA_EXPORT_FUNC(void)
numba_rnd_init(rnd_state_t *state, unsigned int seed)
{
    unsigned int pos;
    seed &= 0xffffffffU;

    /* Knuth's PRNG as used in the Mersenne Twister reference implementation */
    for (pos = 0; pos < MT_N; pos++) {
        state->mt[pos] = seed;
        seed = (1812433253U * (seed ^ (seed >> 30)) + pos + 1) & 0xffffffffU;
    }
    state->index = MT_N;
    state->has_gauss = 0;
    state->gauss = 0.0;
}

/* Perturb mt[] with a key array */
static void
rnd_init_by_array(rnd_state_t *state, unsigned int init_key[], size_t key_length)
{
    size_t i, j, k;
    unsigned int *mt = state->mt;

    numba_rnd_init(state, 19650218U);
    i = 1; j = 0;
    k = (MT_N > key_length ? MT_N : key_length);
    for (; k; k--) {
        mt[i] = (mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 30)) * 1664525U))
                 + init_key[j] + (unsigned int) j; /* non linear */
        mt[i] &= 0xffffffffU;
        i++; j++;
        if (i >= MT_N) { mt[0] = mt[MT_N - 1]; i = 1; }
        if (j >= key_length) j = 0;
    }
    for (k = MT_N - 1; k; k--) {
        mt[i] = (mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 30)) * 1566083941U))
                 - (unsigned int) i; /* non linear */
        mt[i] &= 0xffffffffU;
        i++;
        if (i >= MT_N) { mt[0] = mt[MT_N - 1]; i=1; }
    }

    mt[0] = 0x80000000U; /* MSB is 1; ensuring non-zero initial array */
    state->index = MT_N;
    state->has_gauss = 0;
    state->gauss = 0.0;
}

/* Random-initialize the given state (for use at startup) */
NUMBA_EXPORT_FUNC(int)
_numba_rnd_random_seed(rnd_state_t *state)
{
    PyObject *timemod, *timeobj;
    double timeval;
    Py_uintptr_t seed;
    unsigned int seed32;
    void *dummy;

    /* XXX we could get a seed using _PyOS_URandom() instead */

    timemod = PyImport_ImportModuleNoBlock("time");
    if (!timemod)
        return -1;
    timeobj = PyObject_CallMethod(timemod, "time", NULL);
    Py_DECREF(timemod);
    timeval = PyFloat_AsDouble(timeobj);
    Py_DECREF(timeobj);
    if (timeval == -1 && PyErr_Occurred())
        return -1;
    /* Mix in seconds and nanoseconds */
    seed = (Py_uintptr_t) timeval ^ (Py_uintptr_t) (timeval * 1e9);
#ifndef _WIN32
    seed ^= getpid();
#endif
    /* Address space randomization bits: mix in various pointers. */
    seed ^= (Py_uintptr_t) &timemod;
    seed += (Py_uintptr_t) &PyObject_CallMethod >> 3;
    seed += (Py_uintptr_t) &rnd_init_by_array;
    dummy = malloc(1);
    free(dummy);
    seed += (Py_uintptr_t) &dummy;

    /* Reduce to 32 bits for Mersenne Twisted seeding */
    seed32 = (unsigned int) (seed ^ (seed >> 16));
    numba_rnd_init(state, seed32);
    return 0;
}

/* Python-exposed helpers for state management */
static int
rnd_state_converter(PyObject *obj, rnd_state_t **state)
{
    *state = (rnd_state_t *) PyLong_AsVoidPtr(obj);
    return (*state != NULL || !PyErr_Occurred());
}

NUMBA_EXPORT_FUNC(PyObject *)
_numba_rnd_shuffle(PyObject *self, PyObject *arg)
{
    rnd_state_t *state;
    if (!rnd_state_converter(arg, &state))
        return NULL;
    numba_rnd_shuffle(state);
    Py_RETURN_NONE;
}

NUMBA_EXPORT_FUNC(PyObject *)
_numba_rnd_set_state(PyObject *self, PyObject *args)
{
    int i, index;
    rnd_state_t *state;
    PyObject *tuplearg, *intlist;

    if (!PyArg_ParseTuple(args, "O&O!:rnd_set_state",
                          rnd_state_converter, &state,
                          &PyTuple_Type, &tuplearg))
        return NULL;
    if (!PyArg_ParseTuple(tuplearg, "iO!", &index, &PyList_Type, &intlist))
        return NULL;
    if (PyList_GET_SIZE(intlist) != MT_N) {
        PyErr_SetString(PyExc_ValueError, "list object has wrong size");
        return NULL;
    }
    state->index = index;
    for (i = 0; i < MT_N; i++) {
        PyObject *v = PyList_GET_ITEM(intlist, i);
        unsigned long x = PyLong_AsUnsignedLong(v);
        if (x == (unsigned long) -1 && PyErr_Occurred())
            return NULL;
        state->mt[i] = (unsigned int) x;
    }
    state->has_gauss = 0;
    state->gauss = 0.0;
    Py_RETURN_NONE;
}

NUMBA_EXPORT_FUNC(PyObject *)
_numba_rnd_get_state(PyObject *self, PyObject *arg)
{
    PyObject *intlist;
    int i;
    rnd_state_t *state;
    if (!rnd_state_converter(arg, &state))
        return NULL;

    intlist = PyList_New(MT_N);
    if (intlist == NULL)
        return NULL;
    for (i = 0; i < MT_N; i++) {
        PyObject *v = PyLong_FromUnsignedLong(state->mt[i]);
        if (v == NULL) {
            Py_DECREF(intlist);
            return NULL;
        }
        PyList_SET_ITEM(intlist, i, v);
    }
    return Py_BuildValue("iN", state->index, intlist);
}

NUMBA_EXPORT_FUNC(PyObject *)
_numba_rnd_seed_with_urandom(PyObject *self, PyObject *args)
{
    rnd_state_t *state;
    Py_buffer buf;
    unsigned int *keys;
    unsigned char *bytes;
    size_t i, nkeys;

    if (!PyArg_ParseTuple(args, "O&s*:rnd_seed",
                          rnd_state_converter, &state, &buf)) {
        return NULL;
    }
    /* Make a copy to avoid alignment issues */
    nkeys = buf.len / sizeof(unsigned int);
    keys = (unsigned int *) PyMem_Malloc(nkeys * sizeof(unsigned int));
    if (keys == NULL) {
        PyBuffer_Release(&buf);
        return NULL;
    }
    bytes = (unsigned char *) buf.buf;
    for (i = 0; i < nkeys; i++, bytes += 4) {
        keys[i] = (bytes[3] << 24) + (bytes[2] << 16) +
                  (bytes[1] << 8) + (bytes[0] << 0);
    }
    PyBuffer_Release(&buf);
    rnd_init_by_array(state, keys, nkeys);
    PyMem_Free(keys);
    Py_RETURN_NONE;
}

NUMBA_EXPORT_FUNC(PyObject *)
_numba_rnd_seed(PyObject *self, PyObject *args)
{
    unsigned int seed;
    rnd_state_t *state;

    if (!PyArg_ParseTuple(args, "O&I:rnd_seed",
                          rnd_state_converter, &state, &seed)) {
        PyErr_Clear();
        return _numba_rnd_seed_with_urandom(self, args);
    }
    numba_rnd_init(state, seed);
    Py_RETURN_NONE;
}

/* Random distribution helpers.
 * Most code straight from Numpy's distributions.c. */

#ifndef M_PI
#define M_PI 3.14159265358979323846264338328
#endif

NUMBA_EXPORT_FUNC(unsigned int)
get_next_int32(rnd_state_t *state)
{
    unsigned int y;

    if (state->index == MT_N) {
        numba_rnd_shuffle(state);
        state->index = 0;
    }
    y = state->mt[state->index++];
    /* Tempering */
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680U;
    y ^= (y << 15) & 0xefc60000U;
    y ^= (y >> 18);
    return y;
}

NUMBA_EXPORT_FUNC(double)
get_next_double(rnd_state_t *state)
{
    double a = get_next_int32(state) >> 5;
    double b = get_next_int32(state) >> 6;
    return (a * 67108864.0 + b) / 9007199254740992.0;
}

NUMBA_EXPORT_FUNC(double)
loggam(double x)
{
    double x0, x2, xp, gl, gl0;
    long k, n;

    static double a[10] = {8.333333333333333e-02,-2.777777777777778e-03,
         7.936507936507937e-04,-5.952380952380952e-04,
         8.417508417508418e-04,-1.917526917526918e-03,
         6.410256410256410e-03,-2.955065359477124e-02,
         1.796443723688307e-01,-1.39243221690590e+00};
    x0 = x;
    n = 0;
    if ((x == 1.0) || (x == 2.0))
    {
        return 0.0;
    }
    else if (x <= 7.0)
    {
        n = (long)(7 - x);
        x0 = x + n;
    }
    x2 = 1.0/(x0*x0);
    xp = 2*M_PI;
    gl0 = a[9];
    for (k=8; k>=0; k--)
    {
        gl0 *= x2;
        gl0 += a[k];
    }
    gl = gl0/x0 + 0.5*log(xp) + (x0-0.5)*log(x0) - x0;
    if (x <= 7.0)
    {
        for (k=1; k<=n; k++)
        {
            gl -= log(x0-1.0);
            x0 -= 1.0;
        }
    }
    return gl;
}


NUMBA_EXPORT_FUNC(int64_t)
numba_poisson_ptrs(rnd_state_t *state, double lam)
{
    /* This method is invoked only if the parameter lambda of this
     * distribution is big enough ( >= 10 ). The algorithm used is
     * described in "Hörmann, W. 1992. 'The Transformed Rejection
     * Method for Generating Poisson Random Variables'.
     * The implementation comes straight from Numpy.
     */
    int64_t k;
    double U, V, slam, loglam, a, b, invalpha, vr, us;

    slam = sqrt(lam);
    loglam = log(lam);
    b = 0.931 + 2.53*slam;
    a = -0.059 + 0.02483*b;
    invalpha = 1.1239 + 1.1328/(b-3.4);
    vr = 0.9277 - 3.6224/(b-2);

    while (1)
    {
        U = get_next_double(state) - 0.5;
        V = get_next_double(state);
        us = 0.5 - fabs(U);
        k = (int64_t) floor((2*a/us + b)*U + lam + 0.43);
        if ((us >= 0.07) && (V <= vr))
        {
            return k;
        }
        if ((k < 0) ||
            ((us < 0.013) && (V > us)))
        {
            continue;
        }
        if ((log(V) + log(invalpha) - log(a/(us*us)+b)) <=
            (-lam + (double) k*loglam - loggam((double) k+1)))
        {
            return k;
        }
    }
}

/*
 * Other helpers.
 */

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
numba_cpow(Py_complex *a, Py_complex *b, Py_complex *c) {
    *c = _Py_c_pow(*a, *b);
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
        /* the current code does not handle 0-sized arrays, so give up */
        return 0;
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
    if (cobj == NULL)
        return NULL;
    /* 2.7+ => Cython exports a PyCapsule */
    capsule_name = PyCapsule_GetName(cobj);
    if (capsule_name != NULL) {
        res = PyCapsule_GetPointer(cobj, capsule_name);
    }
    Py_DECREF(cobj);
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

/* Pointer-stuffing functions for tagging a Python list object with an
 * arbitrary pointer.
 * Note a similar hack is used by Python itself, since
 * "list.sort() temporarily sets allocated to -1 to detect mutations".
 */

NUMBA_EXPORT_FUNC(void)
numba_set_list_private_data(PyListObject *listobj, void *ptr)
{
    /* Since ptr is dynamically allocated, it is at least
     * 4- or 8-byte-aligned, meaning we can shift it by a couple bits
     * to the right without losing information.
     */
    if ((size_t) ptr & 1) {
        /* Should never happen */
        Py_FatalError("Numba_set_list_private_data got an unaligned pointer");
    }
    /* Make the pointer distinguishable by forcing it into a negative
     * number (obj->allocated is normally positive, except when sorting
     * where it's changed to -1).
     */
    listobj->allocated = - (Py_ssize_t) ((size_t) ptr >> 1);
}

NUMBA_EXPORT_FUNC(void *)
numba_get_list_private_data(PyListObject *listobj)
{
    if (listobj->allocated < -1) {
        /* A Numba pointer is stuffed in the list, return it */
        return (void *) ((size_t) -listobj->allocated << 1);
    }
    return NULL;
}

NUMBA_EXPORT_FUNC(void)
numba_reset_list_private_data(PyListObject *listobj)
{
    /* Pretend there is no over-allocation; this should be always correct,
     * if not optimal.
     */
    if (listobj->allocated < -1)
        listobj->allocated = PyList_GET_SIZE(listobj);
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

/* Logic for raising an arbitrary object.  Adapted from CPython's ceval.c.
   This *consumes* a reference count to its argument. */
NUMBA_EXPORT_FUNC(int)
numba_do_raise(PyObject *exc)
{
    PyObject *type = NULL, *value = NULL;

    /* We support the following forms of raise:
       raise
       raise <instance>
       raise <type> */

    if (exc == Py_None) {
        /* Reraise */
        PyThreadState *tstate = PyThreadState_GET();
        PyObject *tb;
        Py_DECREF(exc);
        type = tstate->exc_type;
        value = tstate->exc_value;
        tb = tstate->exc_traceback;
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

    if (PyTuple_CheckExact(exc)) {
        /* A (callable, arguments) tuple. */
        if (!PyArg_ParseTuple(exc, "OO", &type, &value)) {
            Py_DECREF(exc);
            goto raise_error;
        }
        value = PyObject_CallObject(type, value);
        Py_DECREF(exc);
        type = NULL;
        if (value == NULL)
            goto raise_error;
        if (!PyExceptionInstance_Check(value)) {
            PyErr_SetString(PyExc_TypeError,
                            "exceptions must derive from BaseException");
            goto raise_error;
        }
        type = PyExceptionInstance_Class(value);
        Py_INCREF(type);
    }
    else if (PyExceptionClass_Check(exc)) {
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
        Py_DECREF(exc);
        PyErr_SetString(PyExc_TypeError,
                        "exceptions must derive from BaseException");
        goto raise_error;
    }

    PyErr_SetObject(type, value);
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
Define bridge for all math functions
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
