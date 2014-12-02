#include "_pymodule.h"
#include <structmember.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include "_dispatcher.h"


typedef struct DispatcherObject{
    PyObject_HEAD
    /* Holds borrowed references to PyCFunction objects */
    dispatcher_t *dispatcher;
    char can_compile;        /* Can auto compile */
    /* Borrowed references */
    PyObject *firstdef, *fallbackdef, *interpdef;
} DispatcherObject;

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

static int tc_intp;

static
PyObject* init_types(PyObject *self, PyObject *args)
{
    PyObject *tmpobj;
    PyObject* dict = PySequence_Fast_GET_ITEM(args, 0);
    int index = 0;

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

    switch(sizeof(void*)) {
    case 4:
        tc_intp = tc_int32;
        break;
    case 8:
        tc_intp = tc_int64;
        break;
    default:
        PyErr_SetString(PyExc_AssertionError, "sizeof(void*) != {4, 8}");
        return NULL;
    }

    Py_RETURN_NONE;
}

static void
Dispatcher_dealloc(DispatcherObject *self)
{
    dispatcher_del(self->dispatcher);
    Py_TYPE(self)->tp_free((PyObject*)self);
}


static
int
Dispatcher_init(DispatcherObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *tmaddrobj;
    void *tmaddr;
    int argct;
    if (!PyArg_ParseTuple(args, "Oi", &tmaddrobj, &argct)) {
        return -1;
    }
    tmaddr = PyLong_AsVoidPtr(tmaddrobj);
    self->dispatcher = dispatcher_new(tmaddr, argct);
    self->can_compile = 1;
    self->firstdef = NULL;
    self->fallbackdef = NULL;
    self->interpdef = NULL;
    return 0;
}


static
PyObject*
Dispatcher_Insert(DispatcherObject *self, PyObject *args)
{
    PyObject *sigtup, *cfunc;
    int i, sigsz;
    int *sig;
    int objectmode = 0;
    int interpmode = 0;

    if (!PyArg_ParseTuple(args, "OO|ii", &sigtup,
                          &cfunc, &objectmode, &interpmode)) {
        return NULL;
    }

    if (!interpmode && !PyObject_TypeCheck(cfunc, &PyCFunction_Type) ) {
        PyErr_SetString(PyExc_TypeError, "must be builtin_function_or_method");
        return NULL;
    }

    sigsz = PySequence_Fast_GET_SIZE(sigtup);
    sig = malloc(sigsz * sizeof(int));

    for (i = 0; i < sigsz; ++i) {
        sig[i] = PyLong_AsLong(PySequence_Fast_GET_ITEM(sigtup, i));
    }

    if (!interpmode) {
        /* The reference to cfunc is borrowed; this only works because the
           derived Python class also stores an (owned) reference to cfunc. */
        dispatcher_add_defn(self->dispatcher, sig, (void*) cfunc);

        /* Add first definition */
        if (!self->firstdef) {
            self->firstdef = cfunc;
        }
    }
    /* Add pure python fallback */
    if (!self->fallbackdef && objectmode){
        self->fallbackdef = cfunc;
    }
    /* Add interpeter fallback */
    if (!self->interpdef && interpmode) {
        self->interpdef = cfunc;
    }

    free(sig);

    Py_RETURN_NONE;
}

static PyObject *str_typeof_pyval = NULL;

/* For void types, we want to keep a reference to the returned type object so
   that it cannot be deleted, and our cache of void types remains valid. So we
   provide the option to do this in retain_reference */
static
int _typecode_fallback(DispatcherObject *dispatcher, PyObject *val,
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
    if (!retain_reference)
        Py_DECREF(tmptype);
    if (tmpcode == NULL)
        return -1;
    typecode = PyLong_AsLong(tmpcode);
    Py_DECREF(tmpcode);
    return typecode;
}

/* Variations on _typecode_fallback for convenience */

static
int typecode_fallback(DispatcherObject *dispatcher, PyObject *val) {
    return _typecode_fallback(dispatcher, val, 0);
}

static
int typecode_fallback_keep_ref(DispatcherObject *dispatcher, PyObject *val) {
    return _typecode_fallback(dispatcher, val, 1);
}

#define N_DTYPES 12
#define N_NDIM 5    /* Fast path for up to 5D array */
#define N_LAYOUT 3
static int cached_arycode[N_NDIM][N_LAYOUT][N_DTYPES];

int dtype_num_to_typecode(int type_num) {
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
        dtype = -1;
    }
    return dtype;
}


static
int typecode_ndarray(DispatcherObject *dispatcher, PyArrayObject *ary) {
    int typecode;
    int dtype;
    int ndim = PyArray_NDIM(ary);
    int layout = 0;

    if (PyArray_ISFARRAY(ary)) {
        layout = 1;
    } else if (PyArray_ISCARRAY(ary)){
        layout = 2;
    }

    if (ndim <= 0 || ndim > N_NDIM) goto FALLBACK;

    dtype = dtype_num_to_typecode(PyArray_TYPE(ary));
    if (dtype == -1) goto FALLBACK;

    /* "Fast" path, using table lookup */
    assert(layout < N_LAYOUT);
    assert(ndim <= N_NDIM);
    assert(dtype < N_DTYPES);

    typecode = cached_arycode[ndim - 1][layout][dtype];
    if (typecode == -1) {
        /* First use of this table entry, so it requires populating */
        typecode = typecode_fallback(dispatcher, (PyObject*)ary);
        cached_arycode[ndim - 1][layout][dtype] = typecode;
    }
    return typecode;

FALLBACK:
    /* "Slow" path */

    /* If this isn't a structured array then we can't use the cache */
    if (PyArray_TYPE(ary) != NPY_VOID)
        return typecode_fallback(dispatcher, (PyObject*)ary);

    /* Check type cache */
    typecode = dispatcher_get_ndarray_typecode(ndim, layout,
                                               PyArray_DESCR(ary));
    if (typecode == -1) {
        /* First use of this type, use fallback and populate the cache */
        typecode = typecode_fallback_keep_ref(dispatcher, (PyObject*)ary);
        dispatcher_insert_ndarray_typecode(ndim, layout, PyArray_DESCR(ary),
                                           typecode);
    }
    return typecode;
}

static
int typecode_arrayscalar(DispatcherObject *dispatcher, PyObject* aryscalar) {
    int typecode;
    PyArray_Descr* descr;
    descr = PyArray_DescrFromScalar(aryscalar);
    if (!descr)
        return typecode_fallback(dispatcher, aryscalar);

    if (descr->type_num == NPY_VOID) {
        typecode = dispatcher_get_arrayscalar_typecode(descr);
        if (typecode == -1) {
            /* Resolve through fallback then populate cache */
            typecode = typecode_fallback_keep_ref(dispatcher, aryscalar);
            dispatcher_insert_arrayscalar_typecode(descr, typecode);
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


static
int typecode(DispatcherObject *dispatcher, PyObject *val) {
    PyTypeObject *tyobj = val->ob_type;
    /* This needs to be kept in sync with Dispatcher.typeof_pyval(),
     * otherwise funny things may happen.
     */
    if (tyobj == &PyInt_Type || tyobj == &PyLong_Type)
        return tc_intp;
    else if (tyobj == &PyFloat_Type)
        return tc_float64;
    else if (tyobj == &PyComplex_Type)
        return tc_complex128;
    /* Array scalar handling */
    else if (PyArray_CheckScalar(val)) {
        return typecode_arrayscalar(dispatcher, val);
    }
    /* Array handling */
    else if (tyobj == &PyArray_Type) {
        return typecode_ndarray(dispatcher, (PyArrayObject*)val);
    }

    return typecode_fallback(dispatcher, val);
}

static
void explain_ambiguous(PyObject *dispatcher, PyObject *args, PyObject *kws) {
    PyObject *callback, *result;
    callback = PyObject_GetAttrString(dispatcher, "_explain_ambiguous");
    if (!callback) {
        PyErr_SetString(PyExc_TypeError, "Ambigous overloading");
        return;
    }
    result = PyObject_Call(callback, args, kws);
    if (result != NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                        "_explain_ambiguous must raise an exception");
        Py_DECREF(result);
    }
    Py_XDECREF(callback);
}

/* A custom, fast, inlinable version of PyCFunction_Call() */
static PyObject *
call_cfunc(PyObject *cfunc, PyObject *args, PyObject *kws)
{
    PyCFunctionWithKeywords fn;
    assert(PyCFunction_Check(cfunc));
    assert(PyCFunction_GET_FLAGS(cfunc) == METH_VARARGS | METH_KEYWORDS);
    fn = (PyCFunctionWithKeywords) PyCFunction_GET_FUNCTION(cfunc);
    return fn(PyCFunction_GET_SELF(cfunc), args, kws);
}

static
PyObject*
compile_and_invoke(DispatcherObject *self, PyObject *args, PyObject *kws)
{
    /* Compile a new one */
    PyObject *cfa, *cfunc, *retval;
    cfa = PyObject_GetAttrString((PyObject*)self, "_compile_for_args");
    if (cfa == NULL)
        return NULL;

    /* NOTE: we call the compiled function ourselves instead of
       letting the Python derived class do it.  This is for proper
       behaviour of globals() in jitted functions (issue #476). */
    cfunc = PyObject_Call(cfa, args, kws);
    Py_DECREF(cfa);

    if (cfunc == NULL)
        return NULL;

    if (PyObject_TypeCheck(cfunc, &PyCFunction_Type)) {
        retval = call_cfunc(cfunc, args, kws);
    } else {
        // Re-enter interpreter
        retval = PyObject_Call(cfunc, args, kws);
    }
    Py_DECREF(cfunc);

    return retval;
}

static
PyObject*
Dispatcher_call(DispatcherObject *self, PyObject *args, PyObject *kws)
{
    PyObject *tmptype, *retval = NULL;
    int *tys;
    int argct;
    int i;
    int prealloc[24];
    int matches;
    PyObject *cfunc;

    /* Shortcut for single definition */
    /*if (!self->can_compile && 1 == dispatcher_count(self->dispatcher)){
        fn = self->firstdef;
        return fn(NULL, args, kws);
    }*/

    argct = PySequence_Fast_GET_SIZE(args);

    if (argct < sizeof(prealloc) / sizeof(int))
        tys = prealloc;
    else
        tys = malloc(argct * sizeof(int));

    for (i = 0; i < argct; ++i) {
        tmptype = PySequence_Fast_GET_ITEM(args, i);
        tys[i] = typecode(self, tmptype);
        if (tys[i] == -1) goto CLEANUP;
    }

    /* We only allow unsafe conversions if compilation of new specializations
       has been disabled. */
    cfunc = dispatcher_resolve(self->dispatcher, tys, &matches,
                               !self->can_compile);

    if (matches == 1) {
        /* Definition is found */
        retval = call_cfunc(cfunc, args, kws);
    } else if (matches == 0) {
        /* No matching definition */
        if (self->can_compile) {
            retval = compile_and_invoke(self, args, kws);
        } else if (self->fallbackdef) {
            /* Have object fallback */
            retval = call_cfunc(self->fallbackdef, args, kws);
        } else {
            /* Raise TypeError */
            PyErr_SetString(PyExc_TypeError, "No matching definition");
            retval = NULL;
        }
    } else if (self->can_compile) {
        /* Ambiguous, but are allowed to compile */
        retval = compile_and_invoke(self, args, kws);
    } else {
        /* Ambiguous */
        explain_ambiguous((PyObject*)self, args, kws);
        retval = NULL;
    }

CLEANUP:
    if (tys != prealloc)
        free(tys);

    return retval;
}

static PyMethodDef Dispatcher_methods[] = {
    { "_insert", (PyCFunction)Dispatcher_Insert, METH_VARARGS,
      "insert new definition"},
//    { "_find", (PyCFunction)Dispatcher_Find, METH_VARARGS,
//      "find matching definition and return a tuple of (argtypes, callable)"},
    { NULL },
};

static PyMemberDef Dispatcher_members[] = {
    {"_can_compile", T_BOOL, offsetof(DispatcherObject, can_compile), 0},
    {NULL}  /* Sentinel */
};


static PyTypeObject DispatcherType = {
#if (PY_MAJOR_VERSION < 3)
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
#else
    PyVarObject_HEAD_INIT(NULL, 0)
#endif
    "_dispatcher.Dispatcher",        /*tp_name*/
    sizeof(DispatcherObject),     /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)Dispatcher_dealloc,                         /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    (PyCFunctionWithKeywords)Dispatcher_call, /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "Dispatcher object",       /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    Dispatcher_methods,        /* tp_methods */
    Dispatcher_members,        /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)Dispatcher_init, /* tp_init */
    0,                         /* tp_alloc */
    0,                         /* tp_new */
};


static PyMethodDef ext_methods[] = {
#define declmethod(func) { #func , ( PyCFunction )func , METH_VARARGS , NULL }
    declmethod(init_types),
    { NULL },
#undef declmethod
};


MOD_INIT(_dispatcher) {
    PyObject *m;
    MOD_DEF(m, "_dispatcher", "No docs", ext_methods)
    if (m == NULL)
        return MOD_ERROR_VAL;

    import_array();

    /* initialize cached_arycode to all ones (in bits) */
    memset(cached_arycode, 0xFF, sizeof(cached_arycode));

    str_typeof_pyval = PyString_InternFromString("typeof_pyval");
    if (str_typeof_pyval == NULL)
        return MOD_ERROR_VAL;

    DispatcherType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&DispatcherType) < 0) {
        return MOD_ERROR_VAL;
    }
    Py_INCREF(&DispatcherType);
    PyModule_AddObject(m, "Dispatcher", (PyObject*)(&DispatcherType));

    return MOD_SUCCESS_VAL(m);
}
