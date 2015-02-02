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

/* For Numpy 1.6 */
#ifndef NPY_ARRAY_BEHAVED
    #define NPY_ARRAY_BEHAVED NPY_BEHAVED
#endif

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
    unsigned int mt[MT_N];
    int index;
} rnd_state_t;

static rnd_state_t py_random_state;
static rnd_state_t np_random_state;

/* Some code portions below from CPython's _randommodule.c, some others
   from Numpy's and Jean-Sebastien Roy's randomkit.c. */

static void
Numba_rnd_shuffle(rnd_state_t *state)
{
    int i;
    unsigned int y;

    for (i = 0; i < MT_N - MT_M; i++) {
        y = (state->mt[i] & MT_UPPER_MASK) | (state->mt[i+1] & MT_LOWER_MASK);
        state->mt[i] = state->mt[i+MT_M] ^ (y >> 1) ^ (-(y & 1) & MT_MATRIX_A);
    }
    for (; i < MT_N - 1; i++) {
        y = (state->mt[i] & MT_UPPER_MASK) | (state->mt[i+1] & MT_LOWER_MASK);
        state->mt[i] = state->mt[i+(MT_M-MT_N)] ^ (y >> 1) ^ (-(y & 1) & MT_MATRIX_A);
    }
    y = (state->mt[MT_N - 1] & MT_UPPER_MASK) | (state->mt[0] & MT_LOWER_MASK);
    state->mt[MT_N - 1] = state->mt[MT_M - 1] ^ (y >> 1) ^ (-(y & 1) & MT_MATRIX_A);
}

/* Initialize mt[] with an integer seed */
static void
Numba_rnd_init(rnd_state_t *state, unsigned int seed)
{
    unsigned int pos;
    seed &= 0xffffffffU;

    /* Knuth's PRNG as used in the Mersenne Twister reference implementation */
    for (pos = 0; pos < MT_N; pos++) {
        state->mt[pos] = seed;
        seed = (1812433253U * (seed ^ (seed >> 30)) + pos + 1) & 0xffffffffU;
    }
    state->index = MT_N;
}

static PyObject *
rnd_shuffle(PyObject *self, PyObject *arg)
{
    rnd_state_t *state = (rnd_state_t *) PyLong_AsVoidPtr(arg);
    if (state == NULL && PyErr_Occurred())
        return NULL;
    Numba_rnd_shuffle(state);
    Py_RETURN_NONE;
}

static PyObject *
rnd_set_state(PyObject *self, PyObject *args)
{
    int i, index;
    rnd_state_t *state;
    PyObject *statearg, *tuplearg, *intlist;

    if (!PyArg_ParseTuple(args, "OO!:rnd_set_state",
                          &statearg, &PyTuple_Type, &tuplearg))
        return NULL;
    state = (rnd_state_t *) PyLong_AsVoidPtr(statearg);
    if (state == NULL && PyErr_Occurred())
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
        int x = PyLong_AsUnsignedLong(v);
        if (x == (unsigned long) -1 && PyErr_Occurred())
            return NULL;
        state->mt[i] = x;
    }
    Py_RETURN_NONE;
}

static PyObject *
rnd_get_state(PyObject *self, PyObject *arg)
{
    PyObject *intlist;
    int i;

    rnd_state_t *state = (rnd_state_t *) PyLong_AsVoidPtr(arg);
    if (state == NULL && PyErr_Occurred())
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

static PyObject *
rnd_init(PyObject *self, PyObject *args)
{
    unsigned int seed;
    rnd_state_t *state;
    PyObject *statearg;

    if (!PyArg_ParseTuple(args, "OI:rnd_init", &statearg, &seed))
        return NULL;
    state = (rnd_state_t *) PyLong_AsVoidPtr(statearg);
    if (state == NULL && PyErr_Occurred())
        return NULL;
    Numba_rnd_init(state, seed);
    Py_RETURN_NONE;
}


/*
 * Other helpers.
 */

/* provide 64-bit division function to 32-bit platforms */
static
int64_t Numba_sdiv(int64_t a, int64_t b) {
    return a / b;
}

static
uint64_t Numba_udiv(uint64_t a, uint64_t b) {
    return a / b;
}

/* provide 64-bit remainder function to 32-bit platforms */
static
int64_t Numba_srem(int64_t a, int64_t b) {
    return a % b;
}

static
uint64_t Numba_urem(uint64_t a, uint64_t b) {
    return a % b;
}

/* provide complex power */
static
void Numba_cpow(Py_complex *a, Py_complex *b, Py_complex *c) {
    *c = _Py_c_pow(*a, *b);
}


static
int Numba_complex_adaptor(PyObject* obj, Py_complex *out) {
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
static
void* Numba_extract_record_data(PyObject *recordobj, Py_buffer *pbuf) {
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
static
PyObject* Numba_recreate_record(void *pdata, int size, PyObject *dtype) {
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

/*
 * Fill in the *arystruct* with information from the Numpy array *obj*.
 * *arystruct*'s layout is defined in numba.targets.arrayobj (look
 * for the ArrayTemplate class).
 */

typedef struct {
    PyObject *parent;
    npy_intp nitems;
    npy_intp itemsize;
    void *data;
    npy_intp shape_and_strides[];
} arystruct_t;

static
int Numba_adapt_ndarray(PyObject *obj, arystruct_t* arystruct) {
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

    return 0;
}

static
PyObject* Numba_ndarray_new(int nd,
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

/* We use separate functions for datetime64 and timedelta64, to ensure
 * proper type checking.
 */
static npy_int64
Numba_extract_np_datetime(PyObject *td)
{
    if (!PyArray_IsScalar(td, Datetime)) {
        PyErr_SetString(PyExc_TypeError,
                        "expected a numpy.datetime64 object");
        return -1;
    }
    return PyArrayScalar_VAL(td, Timedelta);
}

static npy_int64
Numba_extract_np_timedelta(PyObject *td)
{
    if (!PyArray_IsScalar(td, Timedelta)) {
        PyErr_SetString(PyExc_TypeError,
                        "expected a numpy.timedelta64 object");
        return -1;
    }
    return PyArrayScalar_VAL(td, Timedelta);
}

static PyObject *
Numba_create_np_datetime(npy_int64 value, int unit_code)
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

static PyObject *
Numba_create_np_timedelta(npy_int64 value, int unit_code)
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

static
double Numba_round_even(double y) {
    double z = round(y);
    if (fabs(y-z) == 0.5) {
        /* halfway between two integers; use round-half-even */
        z = 2.0*round(y / 2.0);
    }
    return z;
}

static
float Numba_roundf_even(float y) {
    float z = roundf(y);
    if (fabsf(y-z) == 0.5) {
        /* halfway between two integers; use round-half-even */
        z = 2.0 * roundf(y / 2.0);
    }
    return z;
}

static
uint64_t Numba_fptoui(double x) {
    /* First cast to signed int of the full width to make sure sign extension
       happens (this can make a difference on some platforms...). */
    return (uint64_t) (int64_t) x;
}

static
uint64_t Numba_fptouif(float x) {
    return (uint64_t) (int64_t) x;
}

static
void Numba_release_record_buffer(Py_buffer *buf)
{
    PyBuffer_Release(buf);
}


static
void Numba_gil_ensure(PyGILState_STATE *state) {
    *state = PyGILState_Ensure();
}

static
void Numba_gil_release(PyGILState_STATE *state) {
    PyGILState_Release(*state);
}

/*
Define bridge for all math functions
*/
#define MATH_UNARY(F, R, A) static R Numba_##F(A a) { return F(a); }
#define MATH_BINARY(F, R, A, B) static R Numba_##F(A a, B b) \
                                       { return F(a, b); }
    #include "mathnames.inc"
#undef MATH_UNARY
#undef MATH_BINARY

/*
Expose all functions
*/

static PyObject *
build_c_helpers_dict(void)
{
    PyObject *dct = PyDict_New();
    if (dct == NULL)
        goto error;

#define _declpointer(name, value) do {                 \
    PyObject *o = PyLong_FromVoidPtr(value);           \
    if (o == NULL) goto error;                         \
    if (PyDict_SetItemString(dct, name, o)) {          \
        Py_DECREF(o);                                  \
        goto error;                                    \
    }                                                  \
    Py_DECREF(o);                                      \
} while (0)

#define declmethod(func) _declpointer(#func, &Numba_##func)

#define declpointer(ptr) _declpointer(#ptr, &ptr)

    declmethod(sdiv);
    declmethod(srem);
    declmethod(udiv);
    declmethod(urem);
    declmethod(cpow);
    declmethod(complex_adaptor);
    declmethod(extract_record_data);
    declmethod(release_record_buffer);
    declmethod(adapt_ndarray);
    declmethod(ndarray_new);
    declmethod(extract_np_datetime);
    declmethod(create_np_datetime);
    declmethod(extract_np_timedelta);
    declmethod(create_np_timedelta);
    declmethod(recreate_record);
    declmethod(round_even);
    declmethod(roundf_even);
    declmethod(fptoui);
    declmethod(fptouif);
    declmethod(gil_ensure);
    declmethod(gil_release);
    declmethod(rnd_shuffle);
    declmethod(rnd_init);

    declpointer(py_random_state);
    declpointer(np_random_state);

#define MATH_UNARY(F, R, A) declmethod(F);
#define MATH_BINARY(F, R, A, B) declmethod(F);
    #include "mathnames.inc"
#undef MATH_UNARY
#undef MATH_BINARY

#undef declmethod
    return dct;
error:
    Py_XDECREF(dct);
    return NULL;
}

static PyMethodDef ext_methods[] = {
    { "rnd_get_state", (PyCFunction) rnd_get_state, METH_O, NULL },
    { "rnd_init", (PyCFunction) rnd_init, METH_VARARGS, NULL },
    { "rnd_set_state", (PyCFunction) rnd_set_state, METH_VARARGS, NULL },
    { "rnd_shuffle", (PyCFunction) rnd_shuffle, METH_O, NULL },
    { NULL },
};


MOD_INIT(_helperlib) {
    PyObject *m;
    MOD_DEF(m, "_helperlib", "No docs", ext_methods)
    if (m == NULL)
        return MOD_ERROR_VAL;

    import_array();

    PyModule_AddObject(m, "c_helpers", build_c_helpers_dict());
    PyModule_AddIntConstant(m, "long_min", LONG_MIN);
    PyModule_AddIntConstant(m, "long_max", LONG_MAX);
    PyModule_AddIntConstant(m, "py_buffer_size", sizeof(Py_buffer));
    PyModule_AddIntConstant(m, "py_gil_state_size", sizeof(PyGILState_STATE));

    return MOD_SUCCESS_VAL(m);
}
