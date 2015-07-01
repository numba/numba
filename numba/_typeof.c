#include <string.h>
#include <time.h>
#include <assert.h>

#include "_pymodule.h"
#include "_typeof.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

static int tc_int8;
static int tc_int16;
static int tc_int32;
static int tc_int64;
static int tc_uint8;
static int tc_uint16;
static int tc_uint32;
static int tc_uint64;
static int tc_float32;
static int tc_float64;
static int tc_complex64;
static int tc_complex128;
static int BASIC_TYPECODES[12];

static PyObject* typecache;
static PyObject* ndarray_typecache;

static PyObject *str_typeof_pyval = NULL;

/* When we want to cache the type's typecode for later lookup, we need to
   keep a reference to the returned type object so that it cannot be
   deleted. This is because of the following events occurring when first
   using a @jit function for a given set of types:

    1. typecode_fallback requests a new typecode for an arbitrary Python value;
       this implies creating a Numba type object (on the first dispatcher call);
       the typecode cache is then populated.
    2. matching of the typecode list in _dispatcherimpl.cpp fails, since the
       typecode is new.
    3. we have to compile: compile_and_invoke() is called, it will invoke
       Dispatcher_Insert to register the new signature.

   The reference to the Numba type object returned in step 1 is deleted as
   soon as we call Py_DECREF() on it, since we are holding the only
   reference. If this happens and we use the typecode we got to populate the
   cache, then the cache won't ever return the correct typecode, and the
   dispatcher will never successfully match the typecodes with those of
   some already-compiled instance. So we need to make sure that we don't
   call Py_DECREF() on objects whose typecode will be used to populate the
   cache. This is ensured by calling _typecode_fallback with
   retain_reference == 0.

   Note that technically we are leaking the reference, since we do not continue
   to hold a pointer to the type object that we get back from typeof_pyval.
   However, we don't need to refer to it again, we just need to make sure that
   it is never deleted.
*/
static
int _typecode_fallback(PyObject *dispatcher, PyObject *val,
                       int retain_reference) {
    PyObject *tmptype, *tmpcode;
    int typecode;

    // Go back to the interpreter
    tmptype = PyObject_CallMethodObjArgs((PyObject *) dispatcher,
                                         str_typeof_pyval, val, NULL);
    if (!tmptype) {
        return -1;
    }

    tmpcode = PyObject_GetAttrString(tmptype, "_code");
    if (!retain_reference) {
        Py_DECREF(tmptype);
    }
    if (tmpcode == NULL) {
        return -1;
    }
    typecode = PyLong_AsLong(tmpcode);
    Py_DECREF(tmpcode);
    return typecode;
}

/* Variations on _typecode_fallback for convenience */

static
int typecode_fallback(PyObject *dispatcher, PyObject *val) {
    return _typecode_fallback(dispatcher, val, 0);
}

static
int typecode_fallback_keep_ref(PyObject *dispatcher, PyObject *val) {
    return _typecode_fallback(dispatcher, val, 1);
}

/*
 * Direct lookup table for extra-fast typecode resolution of simple array types.
 */

#define N_DTYPES 12
#define N_NDIM 5    /* Fast path for up to 5D array */
#define N_LAYOUT 3
static int cached_arycode[N_NDIM][N_LAYOUT][N_DTYPES];

/* Convert a Numpy dtype number to an internal index into cached_arycode */
static int dtype_num_to_typecode(int type_num) {
    int dtype;
    switch(type_num) {
    case NPY_INT8:
        dtype = 0;
        break;
    case NPY_INT16:
        dtype = 1;
        break;
    case NPY_INT32:
        dtype = 2;
        break;
    case NPY_INT64:
        dtype = 3;
        break;
    case NPY_UINT8:
        dtype = 4;
        break;
    case NPY_UINT16:
        dtype = 5;
        break;
    case NPY_UINT32:
        dtype = 6;
        break;
    case NPY_UINT64:
        dtype = 7;
        break;
    case NPY_FLOAT32:
        dtype = 8;
        break;
    case NPY_FLOAT64:
        dtype = 9;
        break;
    case NPY_COMPLEX64:
        dtype = 10;
        break;
    case NPY_COMPLEX128:
        dtype = 11;
        break;
    default:
        /* Type not included in the global lookup table */
        dtype = -1;
    }
    return dtype;
}

static
int get_cached_typecode(PyArray_Descr* descr) {
    PyObject* tmpobject = PyDict_GetItem(typecache, (PyObject*)descr);
    if (tmpobject == NULL)
        return -1;

    return PyLong_AsLong(tmpobject);
}

static
void cache_typecode(PyArray_Descr* descr, int typecode) {
    PyObject* value = PyLong_FromLong(typecode);
    PyDict_SetItem(typecache, (PyObject*)descr, value);
    Py_DECREF(value);
}

static
PyObject* ndarray_key(int ndim, int layout, PyArray_Descr* descr) {
    PyObject* tmpndim = PyLong_FromLong(ndim);
    PyObject* tmplayout = PyLong_FromLong(layout);
    PyObject* key = PyTuple_Pack(3, tmpndim, tmplayout, descr);
    Py_DECREF(tmpndim);
    Py_DECREF(tmplayout);
    return key;
}

static
int get_cached_ndarray_typecode(int ndim, int layout, PyArray_Descr* descr) {
    PyObject* key = ndarray_key(ndim, layout, descr);
    PyObject *tmpobject = PyDict_GetItem(ndarray_typecache, key);
    if (tmpobject == NULL)
        return -1;

    Py_DECREF(key);
    return PyLong_AsLong(tmpobject);
}

static
void cache_ndarray_typecode(int ndim, int layout, PyArray_Descr* descr,
                            int typecode) {
    PyObject* key = ndarray_key(ndim, layout, descr);
    PyObject* value = PyLong_FromLong(typecode);
    PyDict_SetItem(ndarray_typecache, key, value);
    Py_DECREF(key);
    Py_DECREF(value);
}

static
int typecode_ndarray(PyObject *dispatcher, PyArrayObject *ary) {
    int typecode;
    int dtype;
    int ndim = PyArray_NDIM(ary);
    int layout = 0;

    /* The order in which we check for the right contiguous-ness is important.
       The order must match the order by numba.numpy_support.map_layout.
    */
    if (PyArray_ISCARRAY(ary)){
        layout = 1;
    } else if (PyArray_ISFARRAY(ary)) {
        layout = 2;
    }

    if (ndim <= 0 || ndim > N_NDIM) goto FALLBACK;

    dtype = dtype_num_to_typecode(PyArray_TYPE(ary));
    if (dtype == -1) goto FALLBACK;

    /* Fast path, using direct table lookup */
    assert(layout < N_LAYOUT);
    assert(ndim <= N_NDIM);
    assert(dtype < N_DTYPES);

    typecode = cached_arycode[ndim - 1][layout][dtype];
    if (typecode == -1) {
        /* First use of this table entry, so it requires populating */
        typecode = typecode_fallback_keep_ref(dispatcher, (PyObject*)ary);
        cached_arycode[ndim - 1][layout][dtype] = typecode;
    }
    return typecode;

FALLBACK:
    /* Slower path, for non-trivial array types */

    /* If this isn't a structured array then we can't use the cache */
    if (PyArray_TYPE(ary) != NPY_VOID)
        return typecode_fallback(dispatcher, (PyObject*)ary);

    /* Check type cache */
    typecode = get_cached_ndarray_typecode(ndim, layout, PyArray_DESCR(ary));
    if (typecode == -1) {
        /* First use of this type, use fallback and populate the cache */
        typecode = typecode_fallback_keep_ref(dispatcher, (PyObject*)ary);
        cache_ndarray_typecode(ndim, layout, PyArray_DESCR(ary), typecode);
    }
    return typecode;
}

static
int typecode_arrayscalar(PyObject *dispatcher, PyObject* aryscalar) {
    int typecode;
    PyArray_Descr* descr;
    descr = PyArray_DescrFromScalar(aryscalar);
    if (!descr)
        return typecode_fallback(dispatcher, aryscalar);

    if (descr->type_num == NPY_VOID) {
        typecode = get_cached_typecode(descr);
        if (typecode == -1) {
            /* Resolve through fallback then populate cache */
            typecode = typecode_fallback_keep_ref(dispatcher, aryscalar);
            cache_typecode(descr, typecode);
        }
        Py_DECREF(descr);
        return typecode;
    }

    typecode = dtype_num_to_typecode(descr->type_num);
    Py_DECREF(descr);
    if (typecode == -1)
        return typecode_fallback(dispatcher, aryscalar);
    return BASIC_TYPECODES[typecode];
}

int
typeof_typecode(PyObject *dispatcher, PyObject *val)
{
    PyTypeObject *tyobj = val->ob_type;
    /* This needs to be kept in sync with Dispatcher.typeof_pyval(),
     * otherwise funny things may happen.
     */
    if (tyobj == &PyInt_Type || tyobj == &PyLong_Type)
        return tc_int64;
    else if (tyobj == &PyFloat_Type)
        return tc_float64;
    else if (tyobj == &PyComplex_Type)
        return tc_complex128;
    /* Array scalar handling */
    else if (PyArray_CheckScalar(val)) {
        return typecode_arrayscalar(dispatcher, val);
    }
    /* Array handling */
    else if (PyType_IsSubtype(tyobj, &PyArray_Type)) {
        return typecode_ndarray(dispatcher, (PyArrayObject*)val);
    }

    return typecode_fallback(dispatcher, val);
}

PyObject *
typeof_init(PyObject *self, PyObject *args)
{
    PyObject *tmpobj;
    PyObject* dict = PySequence_Fast_GET_ITEM(args, 0);
    int index = 0;

    /* Initialize Numpy API */
    import_array();

    #define UNWRAP_TYPE(S)                                              \
        if(!(tmpobj = PyDict_GetItemString(dict, #S))) return NULL;     \
        else {  tc_##S = PyLong_AsLong(tmpobj);                         \
                BASIC_TYPECODES[index++] = tc_##S;  }

    UNWRAP_TYPE(int8)
    UNWRAP_TYPE(int16)
    UNWRAP_TYPE(int32)
    UNWRAP_TYPE(int64)

    UNWRAP_TYPE(uint8)
    UNWRAP_TYPE(uint16)
    UNWRAP_TYPE(uint32)
    UNWRAP_TYPE(uint64)

    UNWRAP_TYPE(float32)
    UNWRAP_TYPE(float64)

    UNWRAP_TYPE(complex64)
    UNWRAP_TYPE(complex128)

    #undef UNWRAP_TYPE

    typecache = PyDict_New();
    ndarray_typecache = PyDict_New();
    if (typecache == NULL || ndarray_typecache == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "failed to create type cache");
        return NULL;
    }

    /* initialize cached_arycode to all ones (in bits) */
    memset(cached_arycode, 0xFF, sizeof(cached_arycode));

    str_typeof_pyval = PyString_InternFromString("typeof_pyval");
    if (str_typeof_pyval == NULL)
        return NULL;

    Py_RETURN_NONE;
}
