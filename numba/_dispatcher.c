#define PY_SSIZE_T_CLEAN

#include "_pymodule.h"

#include <structmember.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>
#include "_dispatcher.h"


typedef struct DispatcherObject{
    PyObject_HEAD
    /* Holds borrowed references to PyCFunction objects */
    dispatcher_t *dispatcher;
    char can_compile;        /* Can auto compile */
    /* Borrowed references */
    PyObject *firstdef, *fallbackdef, *interpdef;
    /* Whether to fold named arguments and default values (false for lifted loops)*/
    int fold_args;
    /* Whether the last positional argument is a stararg */
    int has_stararg;
    /* Tuple of argument names */
    PyObject *argnames;
    /* Tuple of default values */
    PyObject *defargs;
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

static PyObject* typecache;
static PyObject* ndarray_typecache;

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

    typecache = PyDict_New();
    ndarray_typecache = PyDict_New();
    if (typecache == NULL || ndarray_typecache == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "failed to create type cache");
        return NULL;
    }

    Py_RETURN_NONE;
}

static int
Dispatcher_traverse(DispatcherObject *self, visitproc visit, void *arg)
{
    Py_VISIT(self->defargs);
    return 0;
}

static void
Dispatcher_dealloc(DispatcherObject *self)
{
    Py_XDECREF(self->argnames);
    Py_XDECREF(self->defargs);
    dispatcher_del(self->dispatcher);
    Py_TYPE(self)->tp_free((PyObject*)self);
}


static int
Dispatcher_init(DispatcherObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *tmaddrobj;
    void *tmaddr;
    int argct;
    int has_stararg = 0;

    if (!PyArg_ParseTuple(args, "OiiO!O!|i", &tmaddrobj, &argct,
                          &self->fold_args,
                          &PyTuple_Type, &self->argnames,
                          &PyTuple_Type, &self->defargs,
                          &has_stararg)) {
        return -1;
    }
    Py_INCREF(self->argnames);
    Py_INCREF(self->defargs);
    tmaddr = PyLong_AsVoidPtr(tmaddrobj);
    self->dispatcher = dispatcher_new(tmaddr, argct);
    self->can_compile = 1;
    self->firstdef = NULL;
    self->fallbackdef = NULL;
    self->interpdef = NULL;
    self->has_stararg = has_stararg;
    return 0;
}

static PyObject *
Dispatcher_clear(DispatcherObject *self, PyObject *args)
{
    dispatcher_clear(self->dispatcher);
    Py_RETURN_NONE;
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

/* For void types, we need to keep a reference to the returned type object so
   that it cannot be deleted. This is because of the following events occurring
   when first using a @jit function for a given set of types:

    1. typecode_fallback requests a new typecode for an arbitrary Python value;
       this implies creating a Numba type object (on the first dispatcher call);
       the typecode cache is then populated.
    2. matching of the typecode list in _dispatcherimpl.cpp fails, since the
       typecode is new.
    3. we have to compile: compile_and_invoke() is called, it will invoke
       Dispatcher_Insert to register the new signature.

   The reference returned in step 1 is deleted as soon as we call Py_DECREF() on
   it, since we are holding the only reference. If this happens and we use the
   typecode we got to populate the cache, then the cache won't ever return the
   correct typecode, and the dispatcher will never successfully match the typecodes
   with those of some already-compiled instance. So we need to make sure that we
   don't call Py_DECREF() on objects whose typecode will be used to populate the
   cache. This is ensured by calling _typecode_fallback with retain_reference ==
   0.

   Note that technically we are leaking the reference, since we do not continue
   to hold a pointer to the type object that we get back from typeof_pyval.
   However, we don't need to refer to it again, we just need to make sure that
   it is never deleted.
*/
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
int typecode_ndarray(DispatcherObject *dispatcher, PyArrayObject *ary) {
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
    typecode = get_cached_ndarray_typecode(ndim, layout, PyArray_DESCR(ary));
    if (typecode == -1) {
        /* First use of this type, use fallback and populate the cache */
        typecode = typecode_fallback_keep_ref(dispatcher, (PyObject*)ary);
        cache_ndarray_typecode(ndim, layout, PyArray_DESCR(ary), typecode);
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
    else if (PyType_IsSubtype(tyobj, &PyArray_Type)) {
        return typecode_ndarray(dispatcher, (PyArrayObject*)val);
    }

    return typecode_fallback(dispatcher, val);
}

static
void explain_issue(PyObject *dispatcher, PyObject *args, PyObject *kws,
                   const char *method_name, const char *default_msg)
{
    PyObject *callback, *result;
    callback = PyObject_GetAttrString(dispatcher, method_name);
    if (!callback) {
        PyErr_SetString(PyExc_TypeError, default_msg);
        return;
    }
    result = PyObject_Call(callback, args, kws);
    Py_DECREF(callback);
    if (result != NULL) {
        PyErr_Format(PyExc_RuntimeError, "%s must raise an exception",
                     method_name);
        Py_DECREF(result);
    }
}

static
void explain_ambiguous(PyObject *dispatcher, PyObject *args, PyObject *kws)
{
    explain_issue(dispatcher, args, kws, "_explain_ambiguous",
                  "Ambigous overloading");
}

static
void explain_matching_error(PyObject *dispatcher, PyObject *args, PyObject *kws)
{
    explain_issue(dispatcher, args, kws, "_explain_matching_error",
                  "No matching definition");
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

static int
find_named_args(DispatcherObject *self, PyObject **pargs, PyObject **pkws)
{
    PyObject *oldargs = *pargs, *newargs;
    PyObject *kws = *pkws;
    Py_ssize_t pos_args = PyTuple_GET_SIZE(oldargs);
    Py_ssize_t named_args, total_args, i;
    Py_ssize_t func_args = PyTuple_GET_SIZE(self->argnames);
    Py_ssize_t defaults = PyTuple_GET_SIZE(self->defargs);
    /* Last parameter with a default value */
    Py_ssize_t last_def = (self->has_stararg)
                          ? func_args - 2
                          : func_args - 1;
    /* First parameter with a default value */
    Py_ssize_t first_def = last_def - defaults + 1;
    /* Minimum number of required arguments */
    Py_ssize_t minargs = first_def;

    if (kws != NULL)
        named_args = PyDict_Size(kws);
    else
        named_args = 0;
    total_args = pos_args + named_args;
    if (!self->has_stararg && total_args > func_args) {
        PyErr_Format(PyExc_TypeError,
                     "too many arguments: expected %d, got %d",
                     (int) func_args, (int) total_args);
        return -1;
    }
    else if (total_args < minargs) {
        if (minargs == func_args)
            PyErr_Format(PyExc_TypeError,
                         "not enough arguments: expected %d, got %d",
                         (int) minargs, (int) total_args);
        else
            PyErr_Format(PyExc_TypeError,
                         "not enough arguments: expected at least %d, got %d",
                         (int) minargs, (int) total_args);
        return -1;
    }
    newargs = PyTuple_New(func_args);
    if (!newargs)
        return -1;
    /* First pack the stararg */
    if (self->has_stararg) {
        Py_ssize_t stararg_size = Py_MAX(0, pos_args - func_args + 1);
        PyObject *stararg = PyTuple_New(stararg_size);
        if (!stararg) {
            Py_DECREF(newargs);
            return -1;
        }
        for (i = 0; i < stararg_size; i++) {
            PyObject *value = PyTuple_GET_ITEM(oldargs, func_args - 1 + i);
            Py_INCREF(value);
            PyTuple_SET_ITEM(stararg, i, value);
        }
        /* Put it in last position */
        PyTuple_SET_ITEM(newargs, func_args - 1, stararg);

    }
    for (i = 0; i < pos_args; i++) {
        PyObject *value = PyTuple_GET_ITEM(oldargs, i);
        if (self->has_stararg && i >= func_args - 1) {
            /* Skip stararg */
            break;
        }
        Py_INCREF(value);
        PyTuple_SET_ITEM(newargs, i, value);
    }

    /* Iterate over missing positional arguments, try to find them in
       named arguments or default values. */
    for (i = pos_args; i < func_args; i++) {
        PyObject *name = PyTuple_GET_ITEM(self->argnames, i);
        if (self->has_stararg && i >= func_args - 1) {
            /* Skip stararg */
            break;
        }
        if (kws != NULL) {
            /* Named argument? */
            PyObject *value = PyDict_GetItem(kws, name);
            if (value != NULL) {
                Py_INCREF(value);
                PyTuple_SET_ITEM(newargs, i, value);
                named_args--;
                continue;
            }
        }
        if (i >= first_def && i <= last_def) {
            /* Argument has a default value? */
            PyObject *value = PyTuple_GET_ITEM(self->defargs, i - first_def);
            Py_INCREF(value);
            PyTuple_SET_ITEM(newargs, i, value);
            continue;
        }
        else if (i < func_args - 1 || !self->has_stararg) {
            PyErr_Format(PyExc_TypeError,
                         "missing argument '%s'",
                         PyString_AsString(name));
            Py_DECREF(newargs);
            return -1;
        }
    }
    if (named_args) {
        PyErr_Format(PyExc_TypeError,
                     "some keyword arguments unexpected");
        Py_DECREF(newargs);
        return -1;
    }
    *pargs = newargs;
    *pkws = NULL;
    return 0;
}

static PyObject*
Dispatcher_call(DispatcherObject *self, PyObject *args, PyObject *kws)
{
    PyObject *tmptype, *retval = NULL;
    int *tys;
    int argct;
    int i;
    int prealloc[24];
    int matches;
    PyObject *cfunc;

    if (self->fold_args) {
        if (find_named_args(self, &args, &kws))
            return NULL;
    }
    else
        Py_INCREF(args);
    /* Now we own a reference to args */

    argct = PySequence_Fast_GET_SIZE(args);

    if (argct < sizeof(prealloc) / sizeof(int))
        tys = prealloc;
    else
        tys = malloc(argct * sizeof(int));

    for (i = 0; i < argct; ++i) {
        tmptype = PySequence_Fast_GET_ITEM(args, i);
        tys[i] = typecode(self, tmptype);
        if (tys[i] == -1)
            goto CLEANUP;
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
            explain_matching_error((PyObject *) self, args, kws);
            retval = NULL;
        }
    } else if (self->can_compile) {
        /* Ambiguous, but are allowed to compile */
        retval = compile_and_invoke(self, args, kws);
    } else {
        /* Ambiguous */
        explain_ambiguous((PyObject *) self, args, kws);
        retval = NULL;
    }

CLEANUP:
    if (tys != prealloc)
        free(tys);
    Py_DECREF(args);

    return retval;
}

static PyMethodDef Dispatcher_methods[] = {
    { "_clear", (PyCFunction)Dispatcher_clear, METH_NOARGS, NULL },
    { "_insert", (PyCFunction)Dispatcher_Insert, METH_VARARGS,
      "insert new definition"},
    { NULL },
};

static PyMemberDef Dispatcher_members[] = {
    {"_can_compile", T_BOOL, offsetof(DispatcherObject, can_compile), 0},
    {NULL}  /* Sentinel */
};


static PyTypeObject DispatcherType = {
#if (PY_MAJOR_VERSION < 3)
    PyObject_HEAD_INIT(NULL)
    0,                                           /* ob_size */
#else
    PyVarObject_HEAD_INIT(NULL, 0)
#endif
    "_dispatcher.Dispatcher",                    /* tp_name */
    sizeof(DispatcherObject),                    /* tp_basicsize */
    0,                                           /* tp_itemsize */
    (destructor)Dispatcher_dealloc,              /* tp_dealloc */
    0,                                           /* tp_print */
    0,                                           /* tp_getattr */
    0,                                           /* tp_setattr */
    0,                                           /* tp_compare */
    0,                                           /* tp_repr */
    0,                                           /* tp_as_number */
    0,                                           /* tp_as_sequence */
    0,                                           /* tp_as_mapping */
    0,                                           /* tp_hash */
    (PyCFunctionWithKeywords)Dispatcher_call,    /* tp_call*/
    0,                                           /* tp_str*/
    0,                                           /* tp_getattro*/
    0,                                           /* tp_setattro*/
    0,                                           /* tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC, /* tp_flags*/
    "Dispatcher object",                         /* tp_doc */
    (traverseproc) Dispatcher_traverse,          /* tp_traverse */
    0,                                           /* tp_clear */
    0,                                           /* tp_richcompare */
    0,                                           /* tp_weaklistoffset */
    0,                                           /* tp_iter */
    0,                                           /* tp_iternext */
    Dispatcher_methods,                          /* tp_methods */
    Dispatcher_members,                          /* tp_members */
    0,                                           /* tp_getset */
    0,                                           /* tp_base */
    0,                                           /* tp_dict */
    0,                                           /* tp_descr_get */
    0,                                           /* tp_descr_set */
    0,                                           /* tp_dictoffset */
    (initproc)Dispatcher_init,                   /* tp_init */
    0,                                           /* tp_alloc */
    0,                                           /* tp_new */
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
